import sys,os
from utils import util
from enviroment import GridWorld
from algorithm import DP

env=GridWorld.GridWorld()
theta,gamma=0.1,0.1
print(env.reset())
print(env.step(1))

def main():
    args=sys.argv[1:]
    if len(args)<=1:
        job_file=args[0] if len(args)==1 else os.path.join('job','experiments.json')
        for spec_file,spec_and_mode in util.read(job_file).items():
            for spec_name,lab_mode in spec_and_mode.items():
                get_spec_and_run(spec_file,spec_name,lab_mode)
    else:
        assert len(args)==3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        get_spec_and_run(*args)

if __name__=='__main__':
    main()