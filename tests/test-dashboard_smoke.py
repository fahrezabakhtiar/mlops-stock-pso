def test_dashboard_import_runs():
    import runpy
    runpy.run_path("app/dashboard.py", run_name="__main__")