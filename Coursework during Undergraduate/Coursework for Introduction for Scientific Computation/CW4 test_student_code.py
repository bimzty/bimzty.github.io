"""
Created in June 2021

@author: pmzejh
"""

import numpy as np
import time
import re
import importlib
import matplotlib
import matplotlib.pyplot as plt
import errno
import sys



############################################################
case_definitions_dict = {
    # Test if R_{4,4} is found correctly
    'Q2a Test': {
            'GeneralCommands': {
                'Import': 'numerical_integration as ni',
                'Input1': 'f = lambda x: np.sin(x)',
                'Input2': 'a = 0; b = np.pi',
                'Input3': 'n = 1; level = 4',
                'Command1': {'cmd': 'ni.romberg_integration(a,b,n,f,level)', 'Output1': 'integral_approx'},
              },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'integral_approx',
                    'ObjectType': (np.float32,np.float64,float)},
            },
    },
    # Testing of errors  
    'Q2b Test': {
            'GeneralCommands': {
                'Import': 'numerical_integration as ni',
                'Input1': 'f = lambda x: 1/x',
                'Input2': 'N = [1,2,4,8]',
                'Input3': 'levels = [1,2,3,4]',
                'Input4': 'a = 1; b = 2',
                'Input5': 'true_val = np.log(2)',
                'Command1': {'cmd': 'ni.compute_errors(N,4,levels,4,f,a,b,true_val)', 'Output1': 'error_matrix', 'Output2': 'fig1'},
                'Command2': {'cmd': 'ni.compute_errors.__doc__','Output1':'doc_string'},
            },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'error_matrix',
                    'ObjectType': (np.ndarray)},
                'Test2: plot' : {
                    'TestObject': 'fig1',
                    'ObjectType': (matplotlib.figure.Figure)},
                'Test3: docstring' : {
                    'TestObject': 'doc_string',
                    'ObjectType': (str)},
                }
             },
    ### This performs the same test as in the template code (diffent style output)
    'Q3 Test - 2-step Adams Bashforth': {
        'GeneralCommands': {
            'Import': 'numerical_ODEs_first_order as no',
            'Input1': 'a = 0; b = 2; ya = 0.5; n = 40',
            'Input2': 'f = lambda t, y: y - t**2 + 1',
            'Input3': 'method = 2',
            'Command1': {'cmd': 'no.adams_bashforth(f, a, b, ya, n, method)', 'Output1': 't','Output2': 'y'},
            'Command2': {'cmd': 't[-4:]', 'Output1': 'final_ts'},
            'Command3': {'cmd': 'y[-4:]', 'Output1': 'final_ys'}
        },
        'Tests' : {
            'Test1' : {
                'TestObject': 'final_ts',
                'ObjectType': (np.ndarray)},
            'Test2' : {
                'TestObject': 'final_ys',
                'ObjectType': (np.ndarray)},
        },
    },
    ### This performs the same test as in the template code (diffent style output)
    'Q4a Test - 2-step Adams Bashforth, second order': {
        'GeneralCommands': {
            'Import': 'numerical_ODEs_second_order as no',
            'Input1': 'a = 0; b = 1; alpha = 0; beta = 1; n = 40',
            'Input2': 'f = lambda t,y0,y1: (2 + np.exp(-t))*np.cos(t)-y0-2*y1',
            'Input3': 'method = 2',
            'Command1': {'cmd': 'no.adams_bashforth_2(f, a, b, alpha, beta, n, method)', 'Output1': 't','Output2': 'y'},
            'Command2': {'cmd': 't[-4:]', 'Output1': 'final_ts'},
            'Command3': {'cmd': 'y[-4:]', 'Output1': 'final_ys'}
        },
        'Tests' : {
            'Test1' : {
                'TestObject': 'final_ts',
                'ObjectType': (np.ndarray)},
            'Test2' : {
                'TestObject': 'final_ys',
                'ObjectType': (np.ndarray)},
        },
    },
    ### This performs the same test as in the template code (diffent style output)
    'Q4b Test - 2-step Adams Bashforth, second order, errors': {
        'GeneralCommands': {
            'Import': 'numerical_ODEs_second_order as no',
            'Input1': 'a = 0; b = 1; alpha = 0; beta = 1; n = 40',
            'Input2': 'f = lambda t,y0,y1: (2 + np.exp(-t))*np.cos(t)-y0-2*y1',
            'Input3': 'true_y = lambda t: np.exp(-t)- np.exp(-t)*np.cos(t) + np.sin(t)',
            'Input4': 'method = 2',
            'Input5': 'no_n = 6; n_vals = 4*2**np.arange(no_n)',
            'Command1': {'cmd': 'no.compute_ode_errors(n_vals,no_n,a,b,alpha,beta,f,true_y)', 'Output1': 'errors'},
            'Command2': {'cmd': 'no.compute_ode_errors.__doc__','Output1':'doc_string'},
        },
        'Tests' : {
            'Test1' : {
                'TestObject': 'errors',
                'ObjectType': (np.ndarray)},
            'Test2: docstring' : {
                'TestObject': 'doc_string',
                'ObjectType': (str)},
        },
    },
}
     


############################################################
def run_test_case(dict_name,case_name,student_command_window):
    
    """
    Runs test case case_name from the dict_name
    It is assumed all files to import are in the current directory
    """

    # Change plt.show() - to avoid pauses when running the code
    plt.show2 = plt.show
    plt.show = lambda x=1: None
    
    student_command_window[case_name] = "<tt>"
    
    key_list = list(dict_name[case_name]["GeneralCommands"].keys())
    
    glob_dict = globals()
    loc_dict = {}


    #Import modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            cmd = "import "+dict_name[case_name]["GeneralCommands"].get(key)

            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:                
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Variable set up (based on Inputs in dictionary)
    for key in key_list:
        if bool(re.search("^Input",key)):
            cmd = dict_name[case_name]["GeneralCommands"].get(key)
            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Initialise the output dictionary
    dict_name[case_name]["Outputs"] = {}
    
    #Run the set of commands
    
    for key in key_list:
            if bool(re.search("^Command",key)):
                #Set up the outputs
                command_key_list = list(dict_name[case_name]["GeneralCommands"][key].keys())
                Outputs = ""
                for cmd_key in command_key_list:
                    if bool(re.search("^Output",cmd_key)):
                        Outputs = Outputs + dict_name[case_name]["GeneralCommands"][key].get(cmd_key) + ", "

                if len(Outputs) >= 3:
                    Outputs = Outputs[0:len(Outputs)-2]

                cmd = Outputs + " = " + dict_name[case_name]["GeneralCommands"][key].get("cmd")

                student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
                try:
                    exec(cmd,glob_dict,loc_dict)
                        
                    #Append the outputs to the Outputs section of the case dictionary
                    for cmd_key in command_key_list:
                        if bool(re.search("^Output",cmd_key)):
                            output_name = dict_name[case_name]["GeneralCommands"][key].get(cmd_key)
                            dict_name[case_name]["Outputs"][output_name] = loc_dict.get(output_name)
                except Exception as e:
                    student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    student_command_window[case_name] = student_command_window[case_name]+'</tt>'

    #Clean up all newly added modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            if str.split(dict_name[case_name]["GeneralCommands"].get(key))[0] in sys.modules.keys(): 
                del sys.modules[str.split(dict_name[case_name]["GeneralCommands"].get(key))[0]] 
    
    # Change back plt.show()
    plt.show = plt.show2

    # Close all open figures
    plt.close('all')

############################################################

def create_html_of_outputs(student_case_dict,cmd_window):
    

    with open('StudentCodeTestOutput.html','w') as file:
        file.writelines('\n'.join(["<!DOCTYPE html>","<html>"]))
        file.write('<head> \n')
        
        #internal stylings
        file.write('<style> \n')
        file.write('p {font-family: sans-serif;}\n')
        file.write('h1 {font-family: sans-serif;}\n')
        file.write('h2 {font-family: sans-serif;}\n')
        file.write('h3 {font-family: sans-serif;}\n')
        file.write('h4 {font-family: sans-serif;}\n')
        file.write('img {max-width: 100%; max-height: auto; object-fit: contain;}\n')
        file.write('th, td {padding-top: 2.5px; padding-bottom: 2.5px; padding-left: 0px; padding-right: 40px;}\n')
        file.writelines(['body {background-color: azure;margin-left: 5%;margin-right: 5%;font-size: 25px;',
          'object-fit: scale-down;overflow-wrap: break-word;}\n'])
        file.writelines(['.command {border: 2px solid black;border-radius: 25px;padding-left: 12px;padding-right: 10px;',
          'padding-top: 5px;padding-bottom: 5px;font-size: 20px;overflow-wrap: break-word;',
          'object-fit: scale-down;background-color: blanchedalmond;}\n'])
        file.write('.space {margin-top: 1cm;}')
        file.writelines(['.test {border: 2px solid black;padding-left: 12px;padding-right: 10px;padding-top: 5px;',
          'padding-bottom: 5px;font-size: 20px;overflow-wrap: break-word;object-fit: scale-down;',
          'background-color: lemonchiffon;}\n'])
        file.writelines(['.output {border: 2px solid black;border-radius: 25px;padding-left: 12px;',
          'padding-right: 10px;padding-top: 5px;padding-bottom: 5px;font-size: 20px;',
          'overflow-wrap: break-word;object-fit: scale-down;background-color: rgb(232, 238, 245);}\n'])
        file.writelines(['pre {white-space: pre-wrap;','word-wrap: break-word;}\n'])
        file.write('</style> \n')
        #end internal stylings

        file.write('Output from Code tests')
        file.write('</head> \n')

        file.write('<body> \n')

        case_keys = student_case_dict.keys()

        for case_name in case_keys:

            file.write('<p> <b>Case: '+case_name+'</b><br></p>\n')
            
            #Output the commands run
            file.write('<div class="command">')
            file.write('<h3> Commands Executed: </h3>\n')
            file.write('<p style="margin-left:60px;"<tt>'+cmd_window[case_name]+'</tt></p>')
            file.write('</div>\n')
            file.write('<div class="space"></div>\n')

            key_list = list(student_case_dict[case_name]["Tests"].keys())

            for key in key_list:
                if bool(re.search("^Test",key)):
                    file.write('<div class="test">')
                    file.write('<h3>'+key+'</h3>\n')

                    test_object_key = student_case_dict[case_name]["Tests"][key].get("TestObject")

                    
                    if "Outputs" in student_case_dict[case_name]:
                            
                        student_output = student_case_dict[case_name]["Outputs"].get(test_object_key)
                        file.write('<div class="output">')
                        file.write('<h4> Student Output: </h4>\n')

                        student_output_type = type(student_output)
                        required_output_type = student_case_dict[case_name]["Tests"][key].get("ObjectType")
                        
                        if not isinstance(student_output,required_output_type):
                            required_string = str(required_output_type).replace("<","&lt")
                            required_string = required_string.replace(">","&gt")
                            received_string = str(student_output_type).replace("<","&lt")
                            received_string = received_string.replace(">","&gt")
                            warn = "requires (one of) <tt>"+required_string+"</tt> received <tt>"+received_string+"</tt>"
                            file.write("<p style=\"margin-left:60px;\"><span style=\"color:red\">Warning</span>: Student output is of incorrect type, "+warn+"</p>\n")

                        file.write('<p style="margin-left:60px;"><tt>'+ test_object_key + ' = </tt></p>') 
                            
                        if isinstance(student_output,matplotlib.figure.Figure):
                            student_output.savefig(test_object_key+".png",bbox_inches = "tight")
                            file.write('<p style="margin-left:90px;"><img src="'+test_object_key+".png\"></p><br><br>\n")
                        else:
                            student_output = str(student_output)
                            
                            #if isinstance(student_output,str):
                            student_output = student_output.replace("<","&lt")
                            student_output = student_output.replace(">","&gt")
                            student_output = student_output.replace('\n','<br>')

                            file.write('<pre><p style="margin-left:90px;">'+ str(student_output) +'</p></pre>')
                        file.write('</div>\n')
                    else:
                        file.write('<div class="output">')
                        file.write('<p style="margin-left:60px;"> Student Output: None </p>\n')
                        file.write('</div>\n')
                    file.write('</div>\n')

                
                        
        file.write('</body> \n')
        file.write('</html> \n')


############################################################
#Test the code and output

case_keys = case_definitions_dict.keys()

student_command_window = {}

for case_name in case_keys:

    print("      Running ",case_name)
    #Run the student code and store output
    run_test_case(case_definitions_dict,case_name,student_command_window)

create_html_of_outputs(case_definitions_dict,student_command_window)
print("      Created file StudentCodeTestOutput.html")
print("          (open it in a web browser)")


