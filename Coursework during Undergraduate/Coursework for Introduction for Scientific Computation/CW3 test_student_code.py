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
    # Test if cubic interpolation is working
    'Q2 Test': {
            'GeneralCommands': {
                'Import': 'approximations as ap',
                'Input1': 'a = 0.5; b = 1.5',
                'Input2': 'p = 2; n = 10',
                'Input3': 'x = np.linspace(a,b,n)',
                'Input4': 'f = lambda x: np.exp(x)+np.sin(np.pi*x)',
                'Command1': {'cmd': 'ap.poly_interpolation(a,b,p,n,x,f,True)', 'Output1': 'interpolant', 'Output2': 'fig1'},
                'Command2': {'cmd': 'np.shape(interpolant)', 'Output1': 'interp_shape'}
              },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'interpolant',
                    'ObjectType': (np.ndarray)},
                'Test2' : {
                    'TestObject': 'fig1',
                    'ObjectType': (matplotlib.figure.Figure)}
            },
    },
    # Testing of errors  
    'Q3 Test - function (a)': {
            'GeneralCommands': {
                'Import': 'compute_errors as ce',
                'Input1': 'a = -1; b = 1',
                'Input2': 'P = np.arange(1,11)',
                'Input3': 'f = lambda x: np.exp(2.0*x)',
                'Command1': {'cmd': 'ce.interpolation_errors(a,b,10,P,f)', 'Output1': 'error_matrix', 'Output2': 'fig2'},
                'Command2': {'cmd': 'ce.interpolation_errors.__doc__','Output1':'doc_string'},
            },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'error_matrix',
                    'ObjectType': (np.ndarray)},
                'Test2: plot' : {
                    'TestObject': 'fig2',
                    'ObjectType': (matplotlib.figure.Figure)},
                'Test3: docstring' : {
                    'TestObject': 'doc_string',
                    'ObjectType': (str)},
                }
             },
    ### This performs the test in the template code
    'Q4 Case I - p = 11 - test in template': {
        'GeneralCommands': {
            'Import': 'approximations as ap',
            'Input1': 'p = 11; n = 4; m = 3',
            'Input2': 'a = 0; b = 1; c = -1; d = 1',
            'Input4': 'x = np.linspace(a,b,n)',
            'Input5': 'y = np.linspace(c,d,m)',
            'Input6': 'X,Y = np.meshgrid(x,y)',
            'Input7': 'f = lambda x,y : np.exp(x**2+y**2)',
            'Command1': {'cmd': 'ap.poly_interpolation_2d(p,a,b,c,d,X,Y,n,m,f,True)', 'Output1': 'interpolant','Output2': 'fig3'},
        },
        'Tests' : {
            'Test1' : {
                'TestObject': 'interpolant',
                'ObjectType': (np.ndarray)},
            'Test2' : {
                'TestObject': 'fig3',
                'ObjectType': (matplotlib.figure.Figure)},
        },
    },
    ### This performs the test in the template code
    'Q5 - p = 3': {
        'GeneralCommands': {
            'Import': 'lagrange_polynomials as lp',
            'Input1': 'p = 3; n = 6',
            'Input2': 'xhat = np.linspace(-0.5,0.5,p+1)',
            'Input3': 'tol = 1.0e-12',
            'Input4': 'x = np.linspace(-0.5,0.5,n)',
            'Command1': {'cmd': 'lp.deriv_lagrange_poly(p,xhat,n,x,tol)', 'Output1': 'dlagrange_matrix','Output2': 'error_flag'},
        },
        'Tests' : {
            'Test1' : {
                'TestObject': 'dlagrange_matrix',
                'ObjectType': (np.ndarray)
            },
            'Test2' : {
                'TestObject': 'error_flag',
                'ObjectType': (int)}
        }
    },
    # Test if linear interpolation differentiation is working
    'Q6 - p = 1': {
            'GeneralCommands': {
                'Import': 'approximations as ap',
                'Input1': 'maxtime = 60',
                'Input2': 'p = 1; h = 0.01',
                'Input3': 'x = 0.5',
                'Input4': 'f = lambda x: np.cos(np.pi*x)+2*x*np.sin(2*np.pi*x);',
                'Command1': {'cmd': 'ap.approximate_derivative(x,p,h,0,f)', 'Output1': 'dapprox0'},
                'Command2': {'cmd': 'ap.approximate_derivative(x,p,h,1,f)', 'Output1': 'dapprox1'},
                },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'dapprox0',
                    'ObjectType': (np.float64,np.float32,float)
                },
                'Test2' : {
                    'TestObject': 'dapprox1',
                    'ObjectType': (np.float64,np.float32,float)
                },
            }
        },
    # Test if code works for part (a)
    'Q7 function (a)': {
            'GeneralCommands': {
                'Import': 'compute_errors as ce',
                'Input1': 'P = np.array([2,4,6,8])',
                'Input2': 'H = np.array([1/4,1/8,1/16,1/32,1/64,1/128,1/256])',
                'Input3': 'x = 1.0',
                'Input4': 'f = lambda x: np.exp(2*x)',
                'Input5': 'fdiff = lambda x: 2*np.exp(2*x)',
                'Command1': {'cmd': 'ce.derivative_errors(x,P,4,H,7,f,fdiff)', 'Output1': 'error_matrix', 'Output2': 'fig4'},
                'Command2': {'cmd': 'ce.derivative_errors.__doc__','Output1':'doc_string'}
            },
            'Tests' : {
                'Test1' : {
                    'TestObject': 'error_matrix',
                    'ObjectType': (np.ndarray),
                },
                'Test2: plot' : {
                    'TestObject': 'fig4',
                    'ObjectType': (matplotlib.figure.Figure)
                }, 
                'Test3' : {
                    'TestObject': 'doc_string',
                    'ObjectType': (str)
                },  
            }   
    }
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


