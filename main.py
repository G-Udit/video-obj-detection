
from yolo_test_video import *
import boto3
import pandas as pd
import sys
import os

session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

S3 = session.resource("s3")

def download_files_from_s3(bucket_name:str,file_key:str):
    local_filename = file_key.split("/")[-1]
    bucket = S3.Bucket(bucket_name)
    try:
        bucket.download_file(file_key, local_filename)
        print(f"{local_filename} downloaded successfully")
    except Exception as e:
        print("Download failed because of:",Exception)

    return local_filename

def upload_files_to_s3(bucket_name:str,file_key:str,local_filename:str):
    bucket = S3.Bucket(bucket_name)
    try:
        bucket.upload_file(local_filename,file_key)
        print(f"{file_key} uploaded successfully")
    except Exception as e:
        print("Upload failed because of:",Exception)



if __name__=="__main__":
    #bucket_name = sys.argv[1]
    #file_key = sys.argv[2]

    bucket_name = "video-analytics-avaulti"
    file_key = "video object detection/9465999-sd_540_960_30fps.mp4"
    local_filename = download_files_from_s3(bucket_name,file_key)

    video_file_path = glob.glob("*mp4*")[0]
    confidence_threshold = 0.5
    with open('labels.txt','r') as f:
        class_dict =  dict(enumerate(f.read().split('\n')))
    
    output_path = "inference_"+video_file_path.split(".")[0]+".avi"
    #print(output_path)
    run_inference(video_file_path,confidence_threshold,class_dict,output_path)
    upload_files_to_s3(bucket_name,file_key=file_key.split("/")[0]+"/"+output_path,local_filename=output_path)

    

