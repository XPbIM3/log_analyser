# What is that:
A simple handy tool to analyze a recorded logs from TunerStudio

## How to use:

as always: 
```
pip install -r Requirements.txt
```
put Your CurrentTune.msq and *.msl logs in input folder and run DataLog.py  
script rely on some initial constans in the begginging of code  -  so check they are valid for Your specific engine
the output is VE_TABLE achieved from analysing AFR miss coefficient during run - will be in output folder

Things to note:  
1)there is hardcoded condition inside script what data to be used:  TPS position >0,  fully warmed,  not in deccelrating state...etc  
2)AFR table will be generated from initial constants
3)Your tune must have a correct injector input lag set. Otherwise this will lead to wrong VE table
