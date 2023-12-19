function varargout = pac94_fy(varargin)
% PAC94_FY MATLAB code for pac94_fy.fig
%      PAC94_FY, by itself, creates a new PAC94_FY or raises the existing
%      singleton*.
%
%      H = PAC94_FY returns the handle to a new PAC94_FY or the handle to
%      the existing singleton*.
%
%      PAC94_FY('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PAC94_FY.M with the given input arguments.
%
%      PAC94_FY('Property','Value',...) creates a new PAC94_FY or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pac94_fy_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pac94_fy_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pac94_fy

% Last Modified by GUIDE v2.5 25-Sep-2017 11:02:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pac94_fy_OpeningFcn, ...
                   'gui_OutputFcn',  @pac94_fy_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before pac94_fy is made visible.
function pac94_fy_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pac94_fy (see VARARGIN)

% Choose default command line output for pac94_fy
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pac94_fy wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pac94_fy_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function Pt1_Callback(hObject, eventdata, handles)
% hObject    handle to Pt1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Pt1 as text
%        str2double(get(hObject,'String')) returns contents of Pt1 as a double


% --- Executes during object creation, after setting all properties.
function Pt1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pt1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Pt2_Callback(hObject, eventdata, handles)
% hObject    handle to Pt2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Pt2 as text
%        str2double(get(hObject,'String')) returns contents of Pt2 as a double


% --- Executes during object creation, after setting all properties.
function Pt2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pt2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Pt3_Callback(hObject, eventdata, handles)
% hObject    handle to Pt3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Pt3 as text
%        str2double(get(hObject,'String')) returns contents of Pt3 as a double


% --- Executes during object creation, after setting all properties.
function Pt3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pt3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Pt4_Callback(hObject, eventdata, handles)
% hObject    handle to Pt4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Pt4 as text
%        str2double(get(hObject,'String')) returns contents of Pt4 as a double


% --- Executes during object creation, after setting all properties.
function Pt4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pt4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
x=eval(get(handles.alpha,'String'));
gamma=eval(get(handles.gamma,'String'));
Fz=eval(get(handles.Fz,'String'));
Fz=Fz./(-1000);
P(18)=str2num(get(handles.A0_4,'String'))
P(1)=str2num(get(handles.A1_4,'String'))
P(2)=str2num(get(handles.A2_4,'String'))
P(3)=str2num(get(handles.A3_4,'String'))
P(4)=str2num(get(handles.A4_4,'String'))
P(5)=str2num(get(handles.A5_4,'String'))
P(6)=str2num(get(handles.A6_4,'String'))
P(7)=str2num(get(handles.A7_4,'String'))
P(8)=str2num(get(handles.A8_4,'String'))
P(9)=str2num(get(handles.A9_4,'String'))
P(10)=str2num(get(handles.A10_4,'String'))
P(11)=str2num(get(handles.A11_4,'String'))
P(12)=str2num(get(handles.A12_4,'String'))
P(13)=str2num(get(handles.A13_4,'String'))
P(14)=str2num(get(handles.A14_4,'String'))
P(15)=str2num(get(handles.A15_4,'String'))
P(16)=str2num(get(handles.A16_4,'String'))
P(17)=str2num(get(handles.A17_4,'String'))
persistent c 
if isempty(c) 
     c=0 
end 
c=c+1; 
color={'r' 'g' 'b'};
result_no_shift=[];
result_shift=[];
wucha=[];
for i=1:length(Fz)
result1=fy_no_shift(P,x,gamma,Fz(i));
result2=fy_shift(P,x,gamma,Fz(i));
axes(handles.axes1)
plot(x,result1,'r',x,result2,'g')
grid on
legend('FyNoShift','FyShift')
title('PAC94@FY')
hold on

axes(handles.axes2)
plot(x,result1-result2)
legend('FyNoShift-FyNoShift')
title('Îó²î')
grid on
hold on

result_no_shift=[result_no_shift;result1];
result_shift=[result_shift;result2];
wucha=[wucha;result1-result2];

end
result_no_shift=[x;result_no_shift]
result_shift=[x;result_shift]
wucha=[x;wucha]
save('jieguo','result_no_shift','result_shift','wucha')

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[FileName,PathName,FilterIndex]=uigetfile('.xlsx')
fullname=[PathName '\' FileName]
para=xlsread(fullname,'Sheet1','a1:c19')
pa=para;
save('pa.mat','pa');
set(handles.Pt1,'String',para(1,1));
set(handles.A0_1,'String',para(2,1));
set(handles.A0_1,'String',para(2,1));
set(handles.A1_1,'String',para(3,1));
set(handles.A2_1,'String',para(4,1));
set(handles.A3_1,'String',para(5,1));
set(handles.A4_1,'String',para(6,1));
set(handles.A5_1,'String',para(7,1));
set(handles.A6_1,'String',para(8,1));
set(handles.A7_1,'String',para(9,1));
set(handles.A8_1,'String',para(10,1));
set(handles.A9_1,'String',para(11,1));
set(handles.A10_1,'String',para(12,1));
set(handles.A11_1,'String',para(13,1));
set(handles.A12_1,'String',para(14,1));
set(handles.A13_1,'String',para(15,1));
set(handles.A14_1,'String',para(16,1));
set(handles.A15_1,'String',para(17,1));
set(handles.A16_1,'String',para(18,1));
set(handles.A17_1,'String',para(19,1));
%---------------------------------------
set(handles.Pt2,'String',para(1,2));
set(handles.A0_2,'String',para(2,2));
set(handles.A1_2,'String',para(3,2));
set(handles.A2_2,'String',para(4,2));
set(handles.A3_2,'String',para(5,2));
set(handles.A4_2,'String',para(6,2));
set(handles.A5_2,'String',para(7,2));
set(handles.A6_2,'String',para(8,2));
set(handles.A7_2,'String',para(9,2));
set(handles.A8_2,'String',para(10,2));
set(handles.A9_2,'String',para(11,2));
set(handles.A10_2,'String',para(12,2));
set(handles.A11_2,'String',para(13,2));
set(handles.A12_2,'String',para(14,2));
set(handles.A13_2,'String',para(15,2));
set(handles.A14_2,'String',para(16,2));
set(handles.A15_2,'String',para(17,2));
set(handles.A16_2,'String',para(18,2));
set(handles.A17_2,'String',para(19,2));
%------------------------------------
set(handles.Pt3,'String',para(1,3));
set(handles.A0_3,'String',para(2,3));
set(handles.A1_3,'String',para(3,3));
set(handles.A2_3,'String',para(4,3));
set(handles.A3_3,'String',para(5,3));
set(handles.A4_3,'String',para(6,3));
set(handles.A5_3,'String',para(7,3));
set(handles.A6_3,'String',para(8,3));
set(handles.A7_3,'String',para(9,3));
set(handles.A8_3,'String',para(10,3));
set(handles.A9_3,'String',para(12,3));
set(handles.A10_3,'String',para(12,3));
set(handles.A11_3,'String',para(13,3));
set(handles.A12_3,'String',para(14,3));
set(handles.A13_3,'String',para(15,3));
set(handles.A14_3,'String',para(16,3));
set(handles.A15_3,'String',para(17,3));
set(handles.A16_3,'String',para(18,3));
set(handles.A17_3,'String',para(19,3));



% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function A11_1_Callback(hObject, eventdata, handles)
% hObject    handle to A11_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A11_1 as text
%        str2double(get(hObject,'String')) returns contents of A11_1 as a double


% --- Executes during object creation, after setting all properties.
function A11_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A11_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A12_1_Callback(hObject, eventdata, handles)
% hObject    handle to A12_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A12_1 as text
%        str2double(get(hObject,'String')) returns contents of A12_1 as a double


% --- Executes during object creation, after setting all properties.
function A12_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A12_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A13_1_Callback(hObject, eventdata, handles)
% hObject    handle to A13_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A13_1 as text
%        str2double(get(hObject,'String')) returns contents of A13_1 as a double


% --- Executes during object creation, after setting all properties.
function A13_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A13_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A14_1_Callback(hObject, eventdata, handles)
% hObject    handle to A14_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A14_1 as text
%        str2double(get(hObject,'String')) returns contents of A14_1 as a double


% --- Executes during object creation, after setting all properties.
function A14_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A14_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A11_2_Callback(hObject, eventdata, handles)
% hObject    handle to A11_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A11_2 as text
%        str2double(get(hObject,'String')) returns contents of A11_2 as a double


% --- Executes during object creation, after setting all properties.
function A11_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A11_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A12_2_Callback(hObject, eventdata, handles)
% hObject    handle to A12_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A12_2 as text
%        str2double(get(hObject,'String')) returns contents of A12_2 as a double


% --- Executes during object creation, after setting all properties.
function A12_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A12_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A13_2_Callback(hObject, eventdata, handles)
% hObject    handle to A13_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A13_2 as text
%        str2double(get(hObject,'String')) returns contents of A13_2 as a double


% --- Executes during object creation, after setting all properties.
function A13_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A13_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A14_2_Callback(hObject, eventdata, handles)
% hObject    handle to A14_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A14_2 as text
%        str2double(get(hObject,'String')) returns contents of A14_2 as a double


% --- Executes during object creation, after setting all properties.
function A14_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A14_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A11_3_Callback(hObject, eventdata, handles)
% hObject    handle to A11_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A11_3 as text
%        str2double(get(hObject,'String')) returns contents of A11_3 as a double


% --- Executes during object creation, after setting all properties.
function A11_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A11_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A12_3_Callback(hObject, eventdata, handles)
% hObject    handle to A12_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A12_3 as text
%        str2double(get(hObject,'String')) returns contents of A12_3 as a double


% --- Executes during object creation, after setting all properties.
function A12_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A12_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A13_3_Callback(hObject, eventdata, handles)
% hObject    handle to A13_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A13_3 as text
%        str2double(get(hObject,'String')) returns contents of A13_3 as a double


% --- Executes during object creation, after setting all properties.
function A13_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A13_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A14_3_Callback(hObject, eventdata, handles)
% hObject    handle to A14_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A14_3 as text
%        str2double(get(hObject,'String')) returns contents of A14_3 as a double


% --- Executes during object creation, after setting all properties.
function A14_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A14_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A11_4_Callback(hObject, eventdata, handles)
% hObject    handle to A11_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A11_4 as text
%        str2double(get(hObject,'String')) returns contents of A11_4 as a double


% --- Executes during object creation, after setting all properties.
function A11_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A11_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A12_4_Callback(hObject, eventdata, handles)
% hObject    handle to A12_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A12_4 as text
%        str2double(get(hObject,'String')) returns contents of A12_4 as a double


% --- Executes during object creation, after setting all properties.
function A12_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A12_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A13_4_Callback(hObject, eventdata, handles)
% hObject    handle to A13_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A13_4 as text
%        str2double(get(hObject,'String')) returns contents of A13_4 as a double


% --- Executes during object creation, after setting all properties.
function A13_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A13_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A14_4_Callback(hObject, eventdata, handles)
% hObject    handle to A14_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A14_4 as text
%        str2double(get(hObject,'String')) returns contents of A14_4 as a double


% --- Executes during object creation, after setting all properties.
function A14_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A14_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A8_1_Callback(hObject, eventdata, handles)
% hObject    handle to A8_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A8_1 as text
%        str2double(get(hObject,'String')) returns contents of A8_1 as a double


% --- Executes during object creation, after setting all properties.
function A8_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A8_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A9_1_Callback(hObject, eventdata, handles)
% hObject    handle to A9_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A9_1 as text
%        str2double(get(hObject,'String')) returns contents of A9_1 as a double


% --- Executes during object creation, after setting all properties.
function A9_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A9_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A10_1_Callback(hObject, eventdata, handles)
% hObject    handle to A10_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A10_1 as text
%        str2double(get(hObject,'String')) returns contents of A10_1 as a double


% --- Executes during object creation, after setting all properties.
function A10_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A10_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A8_2_Callback(hObject, eventdata, handles)
% hObject    handle to A8_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A8_2 as text
%        str2double(get(hObject,'String')) returns contents of A8_2 as a double


% --- Executes during object creation, after setting all properties.
function A8_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A8_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A9_2_Callback(hObject, eventdata, handles)
% hObject    handle to A9_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A9_2 as text
%        str2double(get(hObject,'String')) returns contents of A9_2 as a double


% --- Executes during object creation, after setting all properties.
function A9_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A9_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A10_2_Callback(hObject, eventdata, handles)
% hObject    handle to A10_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A10_2 as text
%        str2double(get(hObject,'String')) returns contents of A10_2 as a double


% --- Executes during object creation, after setting all properties.
function A10_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A10_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A8_3_Callback(hObject, eventdata, handles)
% hObject    handle to A8_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A8_3 as text
%        str2double(get(hObject,'String')) returns contents of A8_3 as a double


% --- Executes during object creation, after setting all properties.
function A8_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A8_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A9_3_Callback(hObject, eventdata, handles)
% hObject    handle to A9_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A9_3 as text
%        str2double(get(hObject,'String')) returns contents of A9_3 as a double


% --- Executes during object creation, after setting all properties.
function A9_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A9_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A10_3_Callback(hObject, eventdata, handles)
% hObject    handle to A10_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A10_3 as text
%        str2double(get(hObject,'String')) returns contents of A10_3 as a double


% --- Executes during object creation, after setting all properties.
function A10_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A10_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A8_4_Callback(hObject, eventdata, handles)
% hObject    handle to A8_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A8_4 as text
%        str2double(get(hObject,'String')) returns contents of A8_4 as a double


% --- Executes during object creation, after setting all properties.
function A8_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A8_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A9_4_Callback(hObject, eventdata, handles)
% hObject    handle to A9_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A9_4 as text
%        str2double(get(hObject,'String')) returns contents of A9_4 as a double


% --- Executes during object creation, after setting all properties.
function A9_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A9_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A10_4_Callback(hObject, eventdata, handles)
% hObject    handle to A10_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A10_4 as text
%        str2double(get(hObject,'String')) returns contents of A10_4 as a double


% --- Executes during object creation, after setting all properties.
function A10_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A10_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A6_1_Callback(hObject, eventdata, handles)
% hObject    handle to A6_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A6_1 as text
%        str2double(get(hObject,'String')) returns contents of A6_1 as a double


% --- Executes during object creation, after setting all properties.
function A6_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A6_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A7_1_Callback(hObject, eventdata, handles)
% hObject    handle to A7_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A7_1 as text
%        str2double(get(hObject,'String')) returns contents of A7_1 as a double


% --- Executes during object creation, after setting all properties.
function A7_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A7_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A16_1_Callback(hObject, eventdata, handles)
% hObject    handle to A16_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A16_1 as text
%        str2double(get(hObject,'String')) returns contents of A16_1 as a double


% --- Executes during object creation, after setting all properties.
function A16_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A16_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A17_1_Callback(hObject, eventdata, handles)
% hObject    handle to A17_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A17_1 as text
%        str2double(get(hObject,'String')) returns contents of A17_1 as a double


% --- Executes during object creation, after setting all properties.
function A17_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A17_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A6_2_Callback(hObject, eventdata, handles)
% hObject    handle to A6_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A6_2 as text
%        str2double(get(hObject,'String')) returns contents of A6_2 as a double


% --- Executes during object creation, after setting all properties.
function A6_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A6_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A7_2_Callback(hObject, eventdata, handles)
% hObject    handle to A7_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A7_2 as text
%        str2double(get(hObject,'String')) returns contents of A7_2 as a double


% --- Executes during object creation, after setting all properties.
function A7_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A7_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A16_2_Callback(hObject, eventdata, handles)
% hObject    handle to A16_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A16_2 as text
%        str2double(get(hObject,'String')) returns contents of A16_2 as a double


% --- Executes during object creation, after setting all properties.
function A16_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A16_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A17_2_Callback(hObject, eventdata, handles)
% hObject    handle to A17_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A17_2 as text
%        str2double(get(hObject,'String')) returns contents of A17_2 as a double


% --- Executes during object creation, after setting all properties.
function A17_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A17_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A6_3_Callback(hObject, eventdata, handles)
% hObject    handle to A6_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A6_3 as text
%        str2double(get(hObject,'String')) returns contents of A6_3 as a double


% --- Executes during object creation, after setting all properties.
function A6_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A6_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A7_3_Callback(hObject, eventdata, handles)
% hObject    handle to A7_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A7_3 as text
%        str2double(get(hObject,'String')) returns contents of A7_3 as a double


% --- Executes during object creation, after setting all properties.
function A7_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A7_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A16_3_Callback(hObject, eventdata, handles)
% hObject    handle to A16_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A16_3 as text
%        str2double(get(hObject,'String')) returns contents of A16_3 as a double


% --- Executes during object creation, after setting all properties.
function A16_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A16_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A17_3_Callback(hObject, eventdata, handles)
% hObject    handle to A17_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A17_3 as text
%        str2double(get(hObject,'String')) returns contents of A17_3 as a double


% --- Executes during object creation, after setting all properties.
function A17_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A17_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A6_4_Callback(hObject, eventdata, handles)
% hObject    handle to A6_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A6_4 as text
%        str2double(get(hObject,'String')) returns contents of A6_4 as a double


% --- Executes during object creation, after setting all properties.
function A6_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A6_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A7_4_Callback(hObject, eventdata, handles)
% hObject    handle to A7_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A7_4 as text
%        str2double(get(hObject,'String')) returns contents of A7_4 as a double


% --- Executes during object creation, after setting all properties.
function A7_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A7_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A16_4_Callback(hObject, eventdata, handles)
% hObject    handle to A16_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A16_4 as text
%        str2double(get(hObject,'String')) returns contents of A16_4 as a double


% --- Executes during object creation, after setting all properties.
function A16_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A16_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A17_4_Callback(hObject, eventdata, handles)
% hObject    handle to A17_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A17_4 as text
%        str2double(get(hObject,'String')) returns contents of A17_4 as a double


% --- Executes during object creation, after setting all properties.
function A17_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A17_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A3_1_Callback(hObject, eventdata, handles)
% hObject    handle to A3_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A3_1 as text
%        str2double(get(hObject,'String')) returns contents of A3_1 as a double


% --- Executes during object creation, after setting all properties.
function A3_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A3_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A4_1_Callback(hObject, eventdata, handles)
% hObject    handle to A4_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A4_1 as text
%        str2double(get(hObject,'String')) returns contents of A4_1 as a double


% --- Executes during object creation, after setting all properties.
function A4_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A4_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A5_1_Callback(hObject, eventdata, handles)
% hObject    handle to A5_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A5_1 as text
%        str2double(get(hObject,'String')) returns contents of A5_1 as a double


% --- Executes during object creation, after setting all properties.
function A5_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A5_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A3_2_Callback(hObject, eventdata, handles)
% hObject    handle to A3_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A3_2 as text
%        str2double(get(hObject,'String')) returns contents of A3_2 as a double


% --- Executes during object creation, after setting all properties.
function A3_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A3_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A4_2_Callback(hObject, eventdata, handles)
% hObject    handle to A4_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A4_2 as text
%        str2double(get(hObject,'String')) returns contents of A4_2 as a double


% --- Executes during object creation, after setting all properties.
function A4_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A4_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A5_2_Callback(hObject, eventdata, handles)
% hObject    handle to A5_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A5_2 as text
%        str2double(get(hObject,'String')) returns contents of A5_2 as a double


% --- Executes during object creation, after setting all properties.
function A5_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A5_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A3_3_Callback(hObject, eventdata, handles)
% hObject    handle to A3_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A3_3 as text
%        str2double(get(hObject,'String')) returns contents of A3_3 as a double


% --- Executes during object creation, after setting all properties.
function A3_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A3_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A4_3_Callback(hObject, eventdata, handles)
% hObject    handle to A4_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A4_3 as text
%        str2double(get(hObject,'String')) returns contents of A4_3 as a double


% --- Executes during object creation, after setting all properties.
function A4_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A4_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A5_3_Callback(hObject, eventdata, handles)
% hObject    handle to A5_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A5_3 as text
%        str2double(get(hObject,'String')) returns contents of A5_3 as a double


% --- Executes during object creation, after setting all properties.
function A5_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A5_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A3_4_Callback(hObject, eventdata, handles)
% hObject    handle to A3_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A3_4 as text
%        str2double(get(hObject,'String')) returns contents of A3_4 as a double


% --- Executes during object creation, after setting all properties.
function A3_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A3_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A4_4_Callback(hObject, eventdata, handles)
% hObject    handle to A4_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A4_4 as text
%        str2double(get(hObject,'String')) returns contents of A4_4 as a double


% --- Executes during object creation, after setting all properties.
function A4_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A4_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A5_4_Callback(hObject, eventdata, handles)
% hObject    handle to A5_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A5_4 as text
%        str2double(get(hObject,'String')) returns contents of A5_4 as a double


% --- Executes during object creation, after setting all properties.
function A5_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A5_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A1_1_Callback(hObject, eventdata, handles)
% hObject    handle to A1_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A1_1 as text
%        str2double(get(hObject,'String')) returns contents of A1_1 as a double


% --- Executes during object creation, after setting all properties.
function A1_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A1_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A2_1_Callback(hObject, eventdata, handles)
% hObject    handle to A2_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A2_1 as text
%        str2double(get(hObject,'String')) returns contents of A2_1 as a double


% --- Executes during object creation, after setting all properties.
function A2_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A2_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A15_1_Callback(hObject, eventdata, handles)
% hObject    handle to A15_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A15_1 as text
%        str2double(get(hObject,'String')) returns contents of A15_1 as a double


% --- Executes during object creation, after setting all properties.
function A15_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A15_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A1_2_Callback(hObject, eventdata, handles)
% hObject    handle to A1_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A1_2 as text
%        str2double(get(hObject,'String')) returns contents of A1_2 as a double


% --- Executes during object creation, after setting all properties.
function A1_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A1_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A2_2_Callback(hObject, eventdata, handles)
% hObject    handle to A2_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A2_2 as text
%        str2double(get(hObject,'String')) returns contents of A2_2 as a double


% --- Executes during object creation, after setting all properties.
function A2_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A2_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A15_2_Callback(hObject, eventdata, handles)
% hObject    handle to A15_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A15_2 as text
%        str2double(get(hObject,'String')) returns contents of A15_2 as a double


% --- Executes during object creation, after setting all properties.
function A15_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A15_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A1_3_Callback(hObject, eventdata, handles)
% hObject    handle to A1_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A1_3 as text
%        str2double(get(hObject,'String')) returns contents of A1_3 as a double


% --- Executes during object creation, after setting all properties.
function A1_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A1_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A2_3_Callback(hObject, eventdata, handles)
% hObject    handle to A2_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A2_3 as text
%        str2double(get(hObject,'String')) returns contents of A2_3 as a double


% --- Executes during object creation, after setting all properties.
function A2_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A2_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A15_3_Callback(hObject, eventdata, handles)
% hObject    handle to A15_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A15_3 as text
%        str2double(get(hObject,'String')) returns contents of A15_3 as a double


% --- Executes during object creation, after setting all properties.
function A15_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A15_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A1_4_Callback(hObject, eventdata, handles)
% hObject    handle to A1_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A1_4 as text
%        str2double(get(hObject,'String')) returns contents of A1_4 as a double


% --- Executes during object creation, after setting all properties.
function A1_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A1_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A2_4_Callback(hObject, eventdata, handles)
% hObject    handle to A2_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A2_4 as text
%        str2double(get(hObject,'String')) returns contents of A2_4 as a double


% --- Executes during object creation, after setting all properties.
function A2_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A2_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A15_4_Callback(hObject, eventdata, handles)
% hObject    handle to A15_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A15_4 as text
%        str2double(get(hObject,'String')) returns contents of A15_4 as a double


% --- Executes during object creation, after setting all properties.
function A15_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A15_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A0_1_Callback(hObject, eventdata, handles)
% hObject    handle to A0_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A0_1 as text
%        str2double(get(hObject,'String')) returns contents of A0_1 as a double


% --- Executes during object creation, after setting all properties.
function A0_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A0_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A0_2_Callback(hObject, eventdata, handles)
% hObject    handle to A0_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A0_2 as text
%        str2double(get(hObject,'String')) returns contents of A0_2 as a double


% --- Executes during object creation, after setting all properties.
function A0_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A0_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A0_3_Callback(hObject, eventdata, handles)
% hObject    handle to A0_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A0_3 as text
%        str2double(get(hObject,'String')) returns contents of A0_3 as a double


% --- Executes during object creation, after setting all properties.
function A0_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A0_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function A0_4_Callback(hObject, eventdata, handles)
% hObject    handle to A0_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of A0_4 as text
%        str2double(get(hObject,'String')) returns contents of A0_4 as a double


% --- Executes during object creation, after setting all properties.
function A0_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to A0_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cal.
function cal_Callback(hObject, eventdata, handles)
% hObject    handle to cal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('pa.mat')
pa1=pa;
pa1(1,:)=[];
y1=pa1(:,1);
y2=pa1(:,2);
y3=pa1(:,3);
x1=pa(1,1);
x2=pa(1,2);
x3=pa(1,3);
ta=str2num(get(handles.Pt4,'String'));
re=nihe(y1,y2,y3,x1,x2,x3,ta);
set(handles.A0_4,'String',re(1));
set(handles.A1_4,'String',re(2));
set(handles.A2_4,'String',re(3));
set(handles.A3_4,'String',re(4));
set(handles.A4_4,'String',re(5));
set(handles.A5_4,'String',re(6));
set(handles.A6_4,'String',re(7));
set(handles.A7_4,'String',re(8));
set(handles.A8_4,'String',re(9));
set(handles.A9_4,'String',re(10));
set(handles.A10_4,'String',re(11));
set(handles.A11_4,'String',re(12));
set(handles.A12_4,'String',re(13));
set(handles.A13_4,'String',re(14));
set(handles.A14_4,'String',re(15));
set(handles.A15_4,'String',re(16));
set(handles.A16_4,'String',re(17));
set(handles.A17_4,'String',re(18));



function alpha_Callback(hObject, eventdata, handles)
% hObject    handle to alpha (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of alpha as text
%        str2double(get(hObject,'String')) returns contents of alpha as a double


% --- Executes during object creation, after setting all properties.
function alpha_CreateFcn(hObject, eventdata, handles)
% hObject    handle to alpha (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gamma_Callback(hObject, eventdata, handles)
% hObject    handle to gamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gamma as text
%        str2double(get(hObject,'String')) returns contents of gamma as a double


% --- Executes during object creation, after setting all properties.
function gamma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Fz_Callback(hObject, eventdata, handles)
% hObject    handle to Fz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Fz as text
%        str2double(get(hObject,'String')) returns contents of Fz as a double


% --- Executes during object creation, after setting all properties.
function Fz_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Fz (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2


% --- Executes on button press in radiobutton3.
function radiobutton3_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton3


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('jieguo.mat')
[filename, pathname] = uiputfile(...
 {'*.xlsx'},...
 'Save as');
% xlswrite(filename,A,sheet,xlRange)
xlswrite(filename,result_no_shift','noshift')
xlswrite(filename,result_shift','shift')
xlswrite(filename,wucha','Îó²î')


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1)
cla
axes(handles.axes2)
cla
