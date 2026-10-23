from project import project
import argparse
import  json
if __name__=="__main__":
    parce=argparse.ArgumentParser(description="Trip Duration Project")  
    c=parce.add_argument("data",help="pass your data to evalute on the model",type=str)
    args=parce.parse_args()
    with open ("json.json","r") as f:
        config=json.load(f)
    x=project(config)
   
    x.evaluate_model(args.data)
