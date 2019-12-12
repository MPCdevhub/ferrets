#!/usr/bin/env bash
# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Hardcoding to us-east-2 as of July 2019 that is the only region that supports Marketplace 
# submissions for ML models
region=${region:-us-east-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email)

# Build the base docker image locally with the image name and then push it to ECR
# with the full name.

#Dockerfile for GPU and CPU are the same here so we can just reference one.
docker build  -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}

echo "Your ECR path for this image is: $fullname"

