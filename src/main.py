import typer
import os
import sys

from utils import get_base_image_tags, run_command, run_tests_in_docker

app = typer.Typer()
SCRIPTS_MOUNT_DIR = "/autogluon/scripts"

@app.command()
def tabular(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "test_tabular"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="test_tabular")
    
@app.command(name="tabular_automm")
def tabular_automm(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "test_tabular"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="test_tabular_automm")

    
@app.command()
def automm(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "test_automm"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="test_automm")
    
@app.command()
def ts(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "test_ts"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="test_ts")
    
@app.command()
def triton(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "test_triton"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="test_triton")
    
@app.command(name="pip_check")
def pip_check(image_tag: str = typer.Option("beta-autogluon-training-cpu:latest", prompt=True)):
    test_module = "pip_check"
    run_tests_in_docker(image_tag, test_module=test_module, test_function_name="pip_check")
    
    
@app.command()
def setup(account_id: str = typer.Option("845660132111", prompt=True),
         region: str = typer.Option("us-west-2", prompt=True),
         local_dlc_repo_dir: str = typer.Option("/workplace/tonyhu/autogluon/deep-learning-containers", prompt=True)):

    # Set environment variables
    os.environ['ACCOUNT_ID'] = account_id
    os.environ['REGION'] = region
    # Change to the DLC local container repo directory
    os.chdir(local_dlc_repo_dir)
    sys.path.append(local_dlc_repo_dir)
    print(f"sys.path: {sys.path}")
    print(f"current working directory: {os.getcwd()}")
    env = os.environ.copy()
    env['PYTHONPATH'] = local_dlc_repo_dir + ':' + env.get('PYTHONPATH', '')
    print(f"env is {env}")

    
    # Install dependencies
    run_command("pip install -r src/requirements.txt", env=None)

    # Login to AWS ECR
    aws_login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    run_command(aws_login_cmd, env=env)

    # Setup the environment
    run_command("bash src/setup.sh autogluon", env=env)

    # Additional ECR login for predefined AWS account
    run_command("aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com")

    # Pull base images
    training_images = get_base_image_tags(local_dlc_repo_dir, 'training')
    inference_images = get_base_image_tags(local_dlc_repo_dir, 'inference')

    for image_tag in training_images + inference_images:
        print(f"Pulling {image_tag}")
        run_command(f"docker pull {image_tag}", env=env)
    
    # build autogluon images using the spec under local_dlc_repo_dir
    for image_type in ["training", "inference"]:
        os.environ["REPOSITORY_NAME"] = f"beta-autogluon-{image_type}"
        for device_type in ["cpu", "gpu"]:
            build_command = f"python src/main.py --buildspec autogluon/{image_type}/buildspec.yml --framework autogluon --image_types {image_type} --device_types {device_type} --py_versions py3"
            run_command(build_command, env=env)
            
            
if __name__ == "__main__":
    app()