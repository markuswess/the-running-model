from ngapp.cli.serve_standalone import main as startup

def main():
    startup(app_module="the_running_model.appconfig")

if __name__ == "__main__":
    main()
