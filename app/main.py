import sys 
sys.path.append('..')

from inference import *
from fastapi import FastAPI, Request

ppi_bp_model = None 
density_model = None 
subgraph = None 

app = FastAPI()

@app.post("/load_models")
async def load_model(info : Request):
    req_info = await info.json()
    global density_model, ppi_bp_model
    model_type = req_info
    print(model_type)
    if(model_type == 'density'):
        density_model = load_density_model()
    elif(model_type == 'ppi'):
        ppi_bp_model = load_ppi_model()

    return {
        "status" : "SUCCESS",
        "data" : req_info
    }

@app.post("/post_subgraph")
async def post_subgraph(info : Request):
    req_info = await info.json()
    global subgraph
    subgraph = req_info
    print(subgraph)
    return {
        "status" : "SUCCESS",
        "data" : req_info
    }

@app.get("/get_ppi_prediction")
def get_ppi_prediction():
    global subgraph, ppi_bp_model
    output = test_ppi_bp_model(ppi_bp_model, subgraph)
    print(output)

    return{
        "output" : output.tolist()[0],
    }

@app.get("/get_density_prediction")
def get_density_prediction():
    global subgraph, density_model
    output = test_density_model(density_model, subgraph)
    print(output)

    return{
        "output" : output.tolist()[0],
    }


@app.get("/get_subgraph")
def show_subgraph():
    global subgraph
    return{
        "subgraph" : subgraph,
    }

@app.get("/test")
def show_subgraph():
    # global subgraph
    return{
        "subgraph" : "test_output",
    }

