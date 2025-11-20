#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

#execute the commands below within the container prior to launch of the inference-service
#swagger setup
echo $MODEL_ID
cp /tmp/swagger/combined.swagger.${MODEL_ID}.json /swagger
cp /swagger/combined.swagger.json /swagger/combined.swagger.archived.json
mv /swagger/combined.swagger.${MODEL_ID}.json /swagger/combined.swagger.json
#add dir to store SSL info
mkdir -p /app/output/tls
#start cron
crond
