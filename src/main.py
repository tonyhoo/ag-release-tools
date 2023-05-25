import typer

app = typer.Typer()

@app.command()
def tabular():
    import test_tabular
    print("Running tabular tests...")
    test_tabular.test_tabular()
    
@app.command()
def automm():
    import test_automm
    print("Running automm tests...")
    test_automm.test_automm
    
@app.command()
def triton():
    import test_triton
    print("Running triton tests...")
    test_triton.test_triton()
    
    
@app.command()
def test_all():
    print("Running all tests...")
    tabular()
    automm()


if __name__ == "__main__":
    app()


