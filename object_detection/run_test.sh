#!/bin/bash

# conda activate tensorflow15

echo "PWD:         ${PWD}"

pytest --html=../tmp/report-test.html
