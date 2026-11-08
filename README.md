# MatEIS – Materials Selection & Ashby Plotter
CSC-214, University of the Punjab, Lahore.

## Run
    pip install -r requirements.txt
    streamlit run app.py
    pytest            # run the test-suite

## Structure
- `engine.py` – database I/O, validation, constraint filters, Ashby indices, Pareto front (no UI code)
- `app.py` – Streamlit interface (constraints, Ashby plot, ranking, database editor)
- `data/materials.json` – 22 materials (metals, ceramics, polymers, composites)
- `test_engine.py` – boundary and invalid-input tests

## Theory
Material index M = f(properties) maximised for a given function/shape/objective
(Ashby). On a log–log chart, M = σy^m/ρ plots as a line of slope 1/m.
