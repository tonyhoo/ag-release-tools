import os
import subprocess
import re
import sys
from collections import defaultdict

from yaml.loader import SafeLoader

from buildspec import Buildspec


class CustomSafeLoader(SafeLoader):
    def ignore_duplicate_anchors(self, *args, **kwargs):
        return None

    # Override the method to avoid storing anchors, thus ignoring duplicates
    def add_anchor(self, *args):
        pass


def get_base_image_tags(local_dlc_repo_dir, section):
    buildspec = Buildspec()  # Create an instance of the Buildspec class
    buildspec_path = os.path.join(
        local_dlc_repo_dir, "autogluon", section, "buildspec.yml"
    )
    buildspec.load(buildspec_path)  # Load the buildspec file

    images_section = buildspec.get("images", {})  # Get the 'images' section

    base_images = []
    for image_info in images_section.values():
        docker_file_path = image_info.get("docker_file")
        if docker_file_path:
            docker_file_full_path = os.path.join(
                local_dlc_repo_dir, "autogluon", section, docker_file_path
            )
            print(f"Checking {docker_file_full_path}")
            try:
                with open(docker_file_full_path, "r") as docker_file:
                    for line in docker_file:
                        line = line.strip()
                        if line.startswith("FROM "):
                            base_image = line.split(" ", 1)[1]  # Extract the image name
                            base_images.append(base_image)
                            break
            except FileNotFoundError:
                print(f"Warning: Docker file not found at {docker_file_full_path}")

    return base_images


def run_command(command, env=None):
    """Run a command using subprocess and raise exception if it fails."""
    print(f"Running '{command}'")
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command '{command}' failed with exit code {result.returncode}: {result.stderr}"
        )


def run_tests_in_docker(image_tag, test_module, test_function_name):
    """Run a specific test function in a Docker container with volume mount."""
    local_repo_path = os.path.dirname(os.path.abspath(__file__))
    container_mount_path = "/autogluon/scripts/"  # The root of the Python package

    # The command to run the test function
    test_command = f"python -c 'from {test_module} import {test_function_name}; {test_function_name}()'"

    print(
        f"Running {test_function_name} from {test_module} in Docker container for image {image_tag}"
    )
    print(f"Mounting {local_repo_path} to {container_mount_path} inside the container")

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{local_repo_path}:{container_mount_path}",
                "-w",
                container_mount_path,  # Set the working directory to the package root
                image_tag,
                test_command,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Test {test_function_name} failed in Docker container for image {image_tag}: {e}"
        )
        sys.exit(1)


def get_latest_autogluon_images(repo_name_prefix: str="pr-autogluon"):
    # Run the Docker CLI command to get the list of images
    result = subprocess.run(
        ['docker', 'images'], 
        capture_output=True, 
        text=True
    )

    # Check if the command was successful
    if result.returncode != 0:
        raise Exception("Failed to get docker images: " + result.stderr)

    # Parse the command output
    lines = result.stdout.splitlines()
    images = defaultdict(dict)

    # Define a regular expression pattern to match the repository names and tags
    # This pattern excludes tags ending with 'multistage-common' or 'pre-push'
    pattern = re.compile(r'^(?P<repository>\S+)\s+(?P<tag>[\w.-]+(?<!multistage-common)(?<!pre-push))\s+(?P<image_id>\S+)')

    # Dictionary to keep track of the latest image ID for each category
    latest_images = {}

    # Process the output, skipping the header line
    for line in lines[1:]:
        match = pattern.match(line)
        if match:
            repository = match.group('repository')
            tag = match.group('tag')
            image_id = match.group('image_id')

            # Check for AutoGluon training and inference images
            if f"{repo_name_prefix}-training" in repository or f"{repo_name_prefix}-inference" in repository:
                type_key = 'training' if 'training' in repository else 'inference'
                gpu_or_cpu = 'gpu' if '-gpu-' in tag else 'cpu'
                # Check if this is the first image of its type or if the tag is greater (i.e., newer) than the current latest
                if gpu_or_cpu not in latest_images.get(type_key, {}) or tag > latest_images[type_key][gpu_or_cpu]['tag']:
                    latest_images.setdefault(type_key, {})[gpu_or_cpu] = {'tag': tag, 'image_id': image_id}

    return latest_images
