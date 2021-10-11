#!/bin/bash
for i in {801..1200}
do
   python3 generate_data.py -n $i -v
done
