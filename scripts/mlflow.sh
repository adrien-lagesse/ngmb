#!/bin/bash

HOSTNAME=$(hostname)
rye run mlflow ui  --host $HOSTNAME:5000