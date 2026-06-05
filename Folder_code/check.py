from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY_CUA_BAN")
project = rf.workspace("cdis-naxqr").project("helmet-detection-bon9f")
dataset = project.version(1).download("yolov8")