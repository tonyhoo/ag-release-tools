import test_tabular
import test_automm
import test_triton
import typer

app = typer.Typer()

@app.command()
def tabular():
    print("Running tabular tests...")
    test_tabular.test_tabular()
    
@app.command()
def automm():
    print("Running automm tests...")
    test_automm.test_automm
    
@app.command()
def triton():
    print("Running triton tests...")
    test_triton.test_triton()


if __name__ == "__main__":
    app()


