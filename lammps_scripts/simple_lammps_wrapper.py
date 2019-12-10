#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:28:16 2019

@author: cdt1801
"""

import subprocess as sp
import os

class Lammps :
    
    # classmethod to redefine these is supplied later
    lammps_cmd = "mpirun lmp_mpi"
       
    def __init__(self,input_file="",data_file=None) :
        self.input_file = input_file
        self.data_file = data_file
        self.log_file="lammps.log"
        # When writing new functions be very careful about whether this should be prepended to the .in file, .log file, etc
        self.work_dir = "."
        self.ready = False # Flag if class object has been correctly setup to run a lammps simulation
        self.error_msg = [] # Track error messages for this instance
        
    def print_error(self): print("\n".join(self.error_msg))    
    
    # Run function for lammps
    def run(self,total_attempts=3):
        if self.ready:
            # have a few goes at running lammps in case it has some unknown random failure
            run_attempts = 0
            while run_attempts <= total_attempts :
                # give a message saying which simulation is about to start
                sp.call(['echo','Starting simulation for lammps file '+self.input_file])     
                # run lammps
                full_lammps_cmd = " ".join([self.lammps_cmd,"-log",self.log_file,"-in",self.input_file,"> /dev/null"])
                run = sp.Popen(full_lammps_cmd, shell=True, executable='/bin/bash',cwd=self.work_dir,stderr=sp.PIPE)
                run.wait()
                # check lammps return code
                if run.returncode == 0 :
                    # Assign an output to object
                    break
                # returncode 1 means lammps failed in a controlled manner
                elif run.returncode == 1 :
                    self.error_msg += ["Lammps encountered a known error for input file "+self.input_file]
                    self.error_msg += [run.stderr]
                    sp.call(['echo',"\n".join(self.error_msg)])   
                    break
                # other returncode means sim failed for some other reason
                else :
                    self.error_msg += ["Lammps encountered an unknown error for input file "+self.input_file]
                    run_attempts += 1
        else:
            self.error_msg += ["Not ready to run lammps for this instance (probably no .in or .data file supplied)."]                
    
    # use this to setup lammps instance before running it
    @classmethod
    def setup(cls,input_file,data_file=None):
        if os.path.isfile(input_file) and (not data_file or os.path.isfile(data_file)):
            new_object = cls(input_file,data_file)
            new_object.ready = True
            return new_object
        else :
            new_err_msg = "Lammps .in file or .data file not found. Returning an empty class object.\n"
            new_object = cls()
            new_object.error_msg += [new_err_msg]
            return new_object
    
    # can initialise using a currently existing instance
    @classmethod
    def update(cls,old_instance,name,update_dict={},new_data_file=None):
        # Check if old_instance is actually an instance of the class
        if isinstance(old_instance,Lammps):
            # Create new directory for this one
            new_work_dir = old_instance.work_dir+"/"+name
            os.makedirs(new_work_dir)
            # Check input file for references to "old" .data or potentials files
            # Also check for dict elements
            new_lines = []
            with open(old_instance.work_dir+"/"+old_instance.input_file,'r') as old_file:
                lines = old_file.readlines()
                for line in lines:
                    words = line.split()
                    key = words[0]
                    # check for potential files or .data files
                    if key in ("pair_coeff","read_data"):
                        for i, word in enumerate(words):
                            if os.path.isfile(new_work_dir+"/../"+word):
                                words[i] = "../"+word
                    if key in update_dict:
                        words = [key,update_dict[key]]
                    new_lines += [" ".join(words)+"\n"]
            # Write new input file
            new_input_file = new_work_dir+"/"+old_instance.input_file
            with open(new_input_file,"w+") as new_file:
                new_file.writelines(new_lines)
            # Initialise new class instance
            new_data_file = old_instance.data_file
            new_instance = cls.setup(old_instance.input_file,new_data_file)
            new_instance.work_dir = new_work_dir
            new_instance.log_file = old_instance.log_file
            return new_instance                                
        
    # use this to setup the command that will be used to run lammps
    # NB this will alter the command for ALL instances of the class that exist or will be created
    @classmethod
    def command(cls,cores,lammps_path = "lmp_mpi") :
        cls.lammps_cmd = "mpirun -n "+str(cores)+" "+lammps_path
