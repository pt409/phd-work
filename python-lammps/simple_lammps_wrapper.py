#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:28:16 2019

Code is split into the main class (Lammps) and examples.

@author: Pat Taylor (pt409)
"""

import subprocess as sp
import os
import sys
import numpy as np
from random import sample
from mendeleev import element

################################# CLASS #######################################

class Lammps :
    
    # classmethod to redefine these is supplied later
    lammps_cmd = "mpirun lmp_mpi"
       
    def __init__(self,input_file="",data_file=None) :
        self.input_file = input_file
        self.data_file = data_file
        self.log_file = input_file.replace(".in",".log")
        # When writing new functions be very careful about whether this should be prepended to the .in file, .log file, etc
        self.work_dir = "."
        self.ready = False # Flag if class object has been correctly setup to run a lammps simulation
        self.error_msg = [] # Track error messages for this instance
        
    def print_error(self): print("No errors.\n") if self.error_msg == [] else print("\n".join(["Errors:"]+self.error_msg))

    def input_loc(self): return self.work_dir+"/"+self.input_file
    def data_loc(self): return self.work_dir+"/"+self.data_file
    def log_loc(self): return self.work_dir+"/"+self.log_file    
    
    # Run function for lammps
    def run(self,total_attempts=3,echo_msg=False):
        if self.ready:
            # have a few goes at running lammps in case it has some unknown random failure
            run_attempts = 0
            while run_attempts <= total_attempts :
                # give a message saying which simulation is about to start
                if echo_msg : sp.call(['echo','Starting simulation for lammps file '+self.input_file])     
                # run lammps
                full_lammps_cmd = " ".join([self.lammps_cmd,"-log",self.log_file,"-in",self.input_file,"> /dev/null"])
                run = sp.Popen(full_lammps_cmd, shell=True, executable='/bin/bash',cwd=self.work_dir,stdout=sp.PIPE,stderr=sp.PIPE)
                run.wait()
                stdout, stderr = run.communicate()
                # check lammps return code
                if run.returncode == 0 :
                    # Assign an output to object
                    break
                # returncode 1 means lammps failed in a controlled manner
                elif run.returncode == 1 :
                    self.error_msg += ["Lammps encountered a known error for input file "+self.input_file]
                    self.error_msg += [stdout.decode('ascii')]
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
            if not data_file:
                new_object = cls(input_file.split("/")[-1],data_file)
            else:
                new_object = cls(input_file.split("/")[-1],data_file.split("/")[-1])
            new_object.work_dir = "."+"/".join(input_file.split("/")[:-1])
            new_object.ready = True
            return new_object
        else :
            new_err_msg = "Lammps .in file or .data file not found. Returning an empty class object.\n"
            new_object = cls()
            new_object.error_msg += [new_err_msg]
            return new_object
        
    @classmethod
    def default_setup(cls,script,loc=".",pot="NiAlCoCrMoTiWTa.set"):
        script_lib = {"sfe_setup":"/sfe_scripts/sfe_1.in",
                      "sfe_min":"/sfe_scripts/sfe_2.in",
                      "sfe_step":"/sfe_scripts/sfe_3_restart.in",
                      "apbe_setup":"/sfe_scripts/apbe_1.in",
                      "apbe_min":"/sfe_scripts/apbe_2.in",
                      "apbe_step":"/sfe_scripts/apbe_3_restart.in",
                      "md_setup":"/md_scripts/md_1.in",
                      "md_run":"/md_scripts/md_2.in",
                      "elastic_setup":"/elastic_scripts/elastic_1.in",
                      "elastic_run":"/elastic_scripts/elastic_2.in",
                      "elastic_deform":"/elastic_scripts/elastic_3.in"}
        try:
            desired_file = sys.path[2]+"/lammps_scripts"+script_lib[script] if sys.path[2] != "" else "lammps_scripts"+script_lib[script]
            new_input_file = loc + "/" + script_lib[script].split("/")[-1] if loc != "." else script_lib[script].split("/")[-1]
            sp.call(["mkdir","-p",loc])
            prefix = (loc.count("/")+1*(loc!="."))*"../" # Prefix for potentials location
            with open(desired_file,'r') as default_file:
                default_content = default_file.read()
                updated_content = default_content.replace(pot,sys.path[2]+"/potentials/"+pot) if sys.path[2] != "" else default_content.replace(pot,prefix+"potentials/"+pot)
            with open(new_input_file,"w+") as write_file:
                write_file.write(updated_content)
            new_object = cls.setup(new_input_file)
            new_object.work_dir = loc
            return new_object
        except KeyError: 
            new_object = cls()
            new_object.error_msg += ["Script name supplied does not exist"]
            return new_object
    
    # can initialise using a currently existing instance
    @classmethod
    def based_on_setup(cls,old_instance,new_name,update_dict={},new_data_file=None,new_dir=False):
        # Check if old_instance is actually an instance of the class
        if isinstance(old_instance,Lammps):
            # Create new directory for this one
            old_work_dir = old_instance.work_dir
            new_work_dir = old_work_dir if not new_dir else old_work_dir+"/"+new_name
            sp.call(["mkdir","-p",new_work_dir])
            # Check input file for references to "old" .data or potentials files
            # Also check for dict elements
            # This code is a bit messy to deal with possibility of 2+ word key
            new_lines = []
            with open(old_instance.input_loc(),'r') as old_file:
                lines = old_file.readlines()
                for line in lines:
                    words = line.split()
                    for j,_ in enumerate(words) :
                        key = " ".join(words[:j+1])
                        # check for user specified update keywords
                        if key in update_dict:
                            words = [key]+update_dict[key].split()
                        # check for potential files or .data files
                        if key in ("pair_coeff","read_data"):
                            for i, word in enumerate(words):
                                if os.path.isfile(new_work_dir+'/'+(new_work_dir.count('/')-old_work_dir.count('/'))*'../'+word):
                                    words[i] = (new_work_dir.count('/')-old_work_dir.count('/'))*'../'+word
                        if key in update_dict or key in ("pair_coeff","read_data"):
                            break # Make sure a found keyword in the file isn't double counted whilst searching the whole line
                    new_lines += [" ".join(words)+"\n"]
            # Write new input file
            new_input_file = new_work_dir+"/"+new_name
            with open(new_input_file,"w+") as new_file:
                new_file.writelines(new_lines)
            # Initialise new class instance
            if not new_data_file: 
                new_mvd_data_file = old_instance.data_file
            else:
                new_mvd_data_file = new_data_file.split("/")[-1]
                sp.call(["cp",new_data_file,new_work_dir+"/"+new_mvd_data_file])
            new_instance = cls(input_file=new_name,data_file=new_mvd_data_file)
            new_instance.work_dir = new_work_dir
            new_instance.ready = True
            return new_instance                                
        
    # use this to setup the command that will be used to run lammps
    # NB this will alter the command for ALL instances of the class that exist or will be created
    @classmethod
    def command(cls,cores,lammps_path = "lmp_mpi") :
        cls.lammps_cmd = "mpirun -n "+str(cores)+" "+lammps_path
        
    # Use this to update the .in file of a given object
    def update(self,update_dict,replace_dict={},new_input_name=None):
        # Check input file for references to "old" .data or potentials files
        # Also check for dict elements
        # This code is a bit messy to deal with possibility of 2+ word key
        new_lines = []
        with open(self.input_loc(),'r') as old_file:
            lines = old_file.readlines()
            for line in lines:
                # Check for simple replacements
                for replace_key in replace_dict:
                    line = line.replace(replace_key,replace_dict[replace_key])
                # Check for updates to lammps keywords
                words = line.split()
                for j,_ in enumerate(words) :
                    key = " ".join(words[:j+1])
                    # check for user specified update keywords
                    if key in update_dict:
                        words = [key]+update_dict[key].split()
                        break # Make sure a found keyword in the file isn't double counted whilst searching the whole line
                new_lines += [" ".join(words)+"\n"]
        # Write new input file
        if new_input_name:
            self.input_file = new_input_name
        with open(self.input_loc(),"w+") as new_file:
            new_file.writelines(new_lines)
        
    def read_log(self,thermo_style,np_out=True,incl_step=True):
        # thermo_style is the list of strings which appear before the quantities to extract
        # specify np_out=False to get a non numpy array output
        with open(self.log_loc(),'r') as infile:
            f = infile.readlines()
        
        out_list = [[] for _ in thermo_style]
        if incl_step: out_list += [[]]
        for l in f:
            l = l.split()
            if incl_step:
                if 'Step' in l:
                    out_list[0] += [float(l[l.index('Step')+1])]
            for i,word in enumerate(thermo_style):
                if incl_step: i += 1
                if word in l:
                    out_list[i] += [float(l[l.index(word)+2])]
        
        if np_out :            
            return np.array(out_list)
        else :
            return out_list
        
    # Some alloy specific functions:
    def alloyify(self,*args):
        # *args : dictionaries describing alloy composition
        dataf = self.data_loc()
        N_atoms = 0
        M_types = 0
        atoms_count = 0 # Flag which section of the file contains atomic positions
        atoms_start = 0 # Starting line of Atoms block within the data file.
        masses_flag = 1
        masses_start = 0 # Starting line pf masses block within 
        counts = np.zeros(len(args)) # Want to count the number of atoms of each type
        in_types = [[] for _ in range(len(args))] # Find and store the intitial atom types for each atom
        f=[] # store readlines
        line_count = 0
        with open(dataf,'r') as data_in:
            # Analyse data as it is read in for number of atoms, types, etc
            while True:
                line = data_in.readline()
                if not line: break
                f += [line]
                line_count += 1
                line = line.rstrip().split()
                if not line : continue # skip blank lines
                if '#' in line[0] : continue # skip comments
                if line[-1] == "atoms": N_atoms += int(line[0])
                if line[-2:] == ["atom","types"]: M_types += int(line[0]) # Not used - should check this matches len(*args)
                if line[0] == "Atoms": atoms_count += 1; continue # Flag start of atomic positions listings
                if atoms_count == 1: atoms_start += line_count-1
                if line[0] == "Masses": masses_flag = 1; continue
                if masses_flag == 1: masses_start += line_count-1; masses_flag = 0
                if atoms_count and atoms_count<=N_atoms:
                    atom_type = int(line[1])
                    counts[atom_type-1] += 1
                    in_types[atom_type-1] += [int(line[0])]
                    atoms_count += 1
           
        # Find all the atom types suplied
        elements = {}
        for _ in args: elements.update(_)
        elements = sorted(elements.keys())
        # Create list of atom types and assigned positions according to number of atoms of each type just found
        # ... and supplied *args (dictionaries for alloy compositions)
        final_dict = {}
        for N,inds,composition in zip(counts,in_types,args):
            for key in composition: composition[key] = int(round(N*composition[key]))
            composition[max(composition)] += int(N-sum(composition.values())) # make sure all atoms are accounted for
            # Track which atom indices have yet to be assigned an element.
            unassigned_inds = set(inds)
            for el in elements:
                type_num = elements.index(el)+1 # Actually assign types by number not element
                selection = set(sample(unassigned_inds,composition.get(el,0)))
                unassigned_inds -= selection
                for i in selection: final_dict[i] = type_num
        
        # Can now write out a new data file 
        self.data_file = "alloyified_"+self.data_file
        with open(self.data_loc(),'w') as data_out:
            for line_count,line in enumerate(f):
                if "atom types" in line: 
                    line = line.split()
                    line[0] = str(len(elements))
                    line = " ".join(line)+"\n"
                if line_count >= atoms_start and line_count < (atoms_start + N_atoms):
                    line = line.split()
                    line[1] = str(final_dict[int(line[0])])
                    line = " ".join(line)+"\n"
                if line_count == masses_start: 
                    line = "\n".join([" ".join([str(i+1),str(element(el).mass)]) for i,el in enumerate(elements)])+"\n"
                if line_count > masses_start and line_count < (masses_start + M_types):
                    continue # Con't write the original masses at all
                data_out.write(line)

################################ EXAMPLES #####################################
                
def alloy_md_properties(composition,name,*args):
    overall_comp = {}
    for sub_comp in composition: overall_comp.update(sub_comp)
    alloy_elements = " ".join(sorted(overall_comp.keys()))
    output_properties = {}
    cubic_sc = 10
    up = 0.002 # Fractional displacements used to calculate lattice params
    # sfe/apbe calculations (very similar)
    def sfe_apbe_protocol(var):
        sfe_dir = name+"/"+var
        sp.call(["mkdir","-p",sfe_dir])
        sfe_setup = Lammps.default_setup(var+"_setup",loc=sfe_dir)
        sfe_setup.run() # This produces a datafile elemental.data
        sfe_min = Lammps.default_setup(var+"_min",loc=sfe_dir)
        sfe_min.data_file = "elemental.data"
        if var == "sfe": 
            sfe_min.alloyify(composition[0].copy(),composition[-1].copy())
        elif var == "apbe":
            sfe_min.alloyify(composition[0].copy(),composition[-1].copy(),composition[0].copy(),composition[-1].copy())
        sfe_min.update(update_dict={"read_data":"alloyified_elemental.data"},replace_dict={"Ni Ni":alloy_elements,"Ni Al Ni Al":alloy_elements})
        sfe_min.run()
        # Find the maximum displacement that will return cell to original equilibrium
        with open(sfe_min.log_loc()) as read_file:
            f = read_file.readlines()
            x_tot = float(f[-17].split()[-2])
        a = x_tot*2/np.sqrt(6) # Lattice parameter
        # Step refers to shift in upper part of supercell to calculate intrinsic stacking fault
        sfe_step = Lammps.default_setup(var+"_step",loc=sfe_dir)
        sfe_step.update(update_dict={"read_data":"alloyified_elemental.data","variable latparam1 equal":str(a)},replace_dict={"Ni Ni":alloy_elements,"Ni Al Ni Al":alloy_elements})
        sfe_step.run()
        with open(sfe_step.log_loc()) as read_file:
            f = read_file.readlines()
            E = float(f[-2].split()[-2])
        return E
    if "sfe" in args:
        output_properties["sfe"] = sfe_apbe_protocol("sfe")
    if "apbe" in args:
        output_properties["apbe"] = sfe_apbe_protocol("apbe")
    
    # md calculations for basic properties i.e. density/lattice parameters
    if "density" in args or "lattice" in args:
        md_dir = name+"/md"
        sp.call(["mkdir","-p",md_dir])
        md_setup = Lammps.default_setup("md_setup",loc=md_dir)
        md_setup.run()
        md_run = Lammps.default_setup("md_run",loc=md_dir)
        md_run.data_file = "elemental.data"
        md_run.alloyify(composition[0].copy(),composition[-1].copy())
        md_run.update(update_dict={"read_data":"alloyified_elemental.data"},replace_dict={"Ni Al":alloy_elements})
        md_run.run()
        dump_values = md_run.read_log(['Density', 'Temp', 'Press', 'Cella', 'Cellb', 'Cellc'])
        density = np.mean(dump_values[1,2:])
        cell_length = np.mean(dump_values[4:,2:])
        lat_param = cell_length/cubic_sc
        output_properties["density"] = density
        output_properties["lattice"] = lat_param
    
    # Elastic constant calculations    
    if "C11" in args or "C12" in args or "C44" in args:
        elastic_dir = name+"/elastic"
        sp.call(["mkdir","-p",elastic_dir])
        elastic_setup = Lammps.default_setup("elastic_setup",loc=elastic_dir)
        if "density" in args or "lattice" in args: 
            elastic_setup.update(update_dict={"variable latparam1 equal":str(lat_param)})
        else: cell_length = 3.52*cubic_sc # default value
        elastic_setup.run()
        # Do NVT simulations for the undisplaced supercell
        elastic_equilib = Lammps.default_setup("elastic_run",loc=elastic_dir)
        elastic_equilib.data_file = "elemental.data"
        elastic_equilib.alloyify(composition[0].copy(),composition[-1].copy())
        elastic_equilib.update(update_dict={"read_data":"alloyified_elemental.data"},replace_dict={"Ni Al":alloy_elements})
        #elastic_equilib.run()
        #dump_values = elastic_equilib.read_log(['Pxx','Pyy','Pzz','Pxy','Pxz','Pyz'])
        #Pij_mean = np.mean(dump_values[1:],axis=1)
        # Now do displaced cell calculations
        directions = []
        C11 = 0 ; C12 = 0 ; C44 = 0
        if "C11" in args or "C12" in args: directions += ["x","y","z"]
        if "C44" in args: directions += ["xy","xz","yz"]
        all_directions = ["x","y","z","xy","xz","yz"]
        for direction in directions:
            for displacement in [-up,up]:
                actual_disp = str(displacement*cell_length)
                change_box_cmd = direction+" delta 0 ${delta} remap units box" if direction in ["x","y","z"] else direction+" delta ${delta} remap units box"
                current_run = Lammps.based_on_setup(elastic_equilib,direction+str(up)+".in",update_dict={"variable delta equal":actual_disp,"change_box all": change_box_cmd})
                current_run.run()
                dump_values = current_run.read_log(['Pxx','Pyy','Pzz','Pxy','Pxz','Pyz'])
                Pij = np.mean(dump_values[1:],axis=1)
                a = all_directions.index(direction) # Voigt index
                if a < 3: 
                    C11 -= Pij[a]/(6*displacement)
                    C12 -= Pij[(a+1)%3]/(12*displacement) + Pij[(a+2)%3]/(12*displacement)
                if a >= 3:
                    C44 -= Pij[a]/(6*displacement)
        if "C11" in args: output_properties["C11"] = C11*1e-4
        if "C12" in args: output_properties["C12"] = C12*1e-4
        if "C44" in args: output_properties["C44"] = C44*1e-4
        
    return tuple(output_properties[key] for key in args)