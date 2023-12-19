function pathname=main_read_all(pathname)
if isempty(pathname)==1
pathname=uigetdir()
end
if pathname==0
    errordlg('请选择数据')
else
hwaitbar=waitbar(0,'数据读取处理中，请稍候...')

path_purefymz=[pathname '\纯侧偏'];
path_purefx=[pathname '\纯纵滑'];
path_combine=[pathname '\复合工况'];
path_l_relaxtion=[pathname '\侧向松弛长度'];
path_z_relaxtion=[pathname '\纵向刚度'];
path_effective=[pathname '\径向刚度'];
path_effective_radius=[pathname '\有效滚动半径'];



file_purefymz=[path_purefymz '\*.xlsx'];
file_purefx=[path_purefx '\*.xlsx'];
file_combine=[path_combine '\*.xlsx'];
file_l_relaxtion=[path_l_relaxtion '\*.xlsx'];
file_z_relaxtion=[path_z_relaxtion '\*.xlsx'];
file_v_stiffness=[path_effective '\*.xlsx'];
file_effective_radius=[path_effective_radius '\*.xlsx'];

f_purefymz=dir(file_purefymz)
f_purefx=dir(file_purefx)
f_combine=dir(file_combine)
f_l_relaxtion=dir(file_l_relaxtion)
f_z_relaxtion=dir(file_z_relaxtion)
f_v_stiffness=dir(file_v_stiffness)
f_effective_radius=dir(file_effective_radius)
%% 

if isempty(f_purefymz)==0
    for i=1:length(f_purefymz)
    file_purefymz_fullname=[path_purefymz '\' f_purefymz(i).name]
%     purefymz=read_new1(file_purefymz_fullname);
    data=read_new1(file_purefymz_fullname);
        if i==1
        purefymz=data
        else
            n=length(data)
            m=length(purefymz)
            for t=1:n
                purefymz(m+t).sr=data(t).sr
                purefymz(m+t).sa=data(t).sa
                purefymz(m+t).ia=data(t).ia
                purefymz(m+t).fx=data(t).fx
                purefymz(m+t).fy=data(t).fy
                purefymz(m+t).fz=data(t).fz
                purefymz(m+t).mz=data(t).mz
                purefymz(m+t).mx=data(t).mx
                purefymz(m+t).name=data(t).name
                purefymz(m+t).ia_flag=data(t).ia_flag
                purefymz(m+t).sa_flag=data(t).sa_flag
                purefymz(m+t).fz_flag=data(t).fz_flag
                purefymz(m+t).filename=data(t).filename
                purefymz(m+t).mode=data(t).mode
            end
        end
      end
    purefymz1=pinjie(purefymz)
else
    purefymz=[];
    purefymz1=[];
%     errordlg('未找到纯侧偏数据')
end 
waitbar(0.25, hwaitbar, ['读取完成' num2str(25) '%'])
%% 

if isempty(f_purefx)==0
    for i=1:length(f_purefx)
    file_purefx_fullname=[path_purefx '\' f_purefx(i).name]

    data=read_new1(file_purefx_fullname);
        if i==1
        purefx=data
        else
            n=length(data)
            m=length(purefx)
            for t=1:n
                purefx(m+t).sr=data(t).sr
                purefx(m+t).sa=data(t).sa
                purefx(m+t).ia=data(t).ia
                purefx(m+t).fx=data(t).fx
                purefx(m+t).fy=data(t).fy
                purefx(m+t).fz=data(t).fz
                purefx(m+t).mz=data(t).mz
                purefx(m+t).mx=data(t).mx
                purefx(m+t).name=data(t).name
                purefx(m+t).ia_flag=data(t).ia_flag
                purefx(m+t).sa_flag=data(t).sa_flag
                purefx(m+t).fz_flag=data(t).fz_flag
                purefx(m+t).filename=data(t).filename
                purefx(m+t).mode=data(t).mode
            end
        end
      end
    purefx1=pinjie(purefx)
else
    purefx=[];
    purefx1=[];
%     errordlg('未找到纯纵滑数据')
end

waitbar(0.5,hwaitbar,['读取完成' num2str(50) '%'])
%% 

if isempty(f_combine)==0
for i=1:length(f_combine)
file_combine_fullname=[path_combine '\' f_combine(i).name]
data=read_new1(file_combine_fullname);
        if i==1
        combine=data
        else
            n=length(data)
            m=length(combine)
            for t=1:n
                combine(m+t).sr=data(t).sr
                combine(m+t).sa=data(t).sa
                combine(m+t).ia=data(t).ia
                combine(m+t).fx=data(t).fx
                combine(m+t).fy=data(t).fy
                combine(m+t).fz=data(t).fz
                combine(m+t).mz=data(t).mz
                combine(m+t).mx=data(t).mx
                combine(m+t).name=data(t).name
                combine(m+t).ia_flag=data(t).ia_flag
                combine(m+t).sa_flag=data(t).sa_flag
                combine(m+t).fz_flag=data(t).fz_flag
                combine(m+t).filename=data(t).filename
                combine(m+t).mode=data(t).mode
            end
        end
end
combine1=pinjie(combine)
else
    combine=[];
    combine1=[];
%     errordlg('未找到联合工况数据')
end
%% 

if isempty(f_l_relaxtion)==0
    file_l_relaxtion_fullname=[path_l_relaxtion '\' f_l_relaxtion.name]
    [status,sheets,xlFormat]= xlsfinfo(file_l_relaxtion_fullname)
    location=ismember(sheets,'Dta1')
    index=find(location,1)
    for i=index:(length(sheets))%寻找位置
        [num,txt,raw]= xlsread(file_l_relaxtion_fullname,sheets{i});
        str=txt{1};


        [row_raw_size,col_raw_size]=size(raw);

        D=ismember(txt,'D Unfiltered - Distance Unfiltered');
        [D_row_chanl,D_col_chanl]=find(D,1);
        Y=ismember(txt,'FYtd Filtered - Lateral Force Filtered');
        [Y_row_chanl,Y_col_chanl]=find(Y,1);
        D=[];
        Y=[];
        for t=9:(row_raw_size)%拼接数据
        D=[D;raw{t,D_col_chanl}]; 
        Y=[Y;raw{t,Y_col_chanl}]; 

        end
        pat = 'FZ((\d*[.]\d*).'
        t = regexp(str, pat, 'tokens')   %气压值
        lateral_relaxtion(i-index+1).data=[D Y];
        lateral_relaxtion(i-index+1).name=str;
        fz=t{1};
        lateral_relaxtion(i-index+1).fz=str2num(fz{1});
  end

    
else
        lateral_relaxtion=[];
%     errordlg('未找到纯侧偏数据')
end 
%% 纵向刚性

if isempty(f_z_relaxtion)==0
    file_z_relaxtion_fullname=[path_z_relaxtion '\' f_z_relaxtion.name]
    [status,sheets,xlFormat]= xlsfinfo(file_z_relaxtion_fullname)
    l_stiffness=xlsread(file_z_relaxtion_fullname,'Sheet1','e90');
    fz=xlsread(file_z_relaxtion_fullname,'Sheet1','c84:g84');
    lo_stiffness=xlsread(file_z_relaxtion_fullname,'Sheet1','c90:g90');
    t=isnan(fz);
    u=isnan(l_stiffness);
    l_stiffness(u)=[];
    fz(t)=[];
    lo_stiffness(t)=[];
    Longitudinal_relaxation.fz=fz;
    Longitudinal_relaxation.lo_stiffness=lo_stiffness;
else
    Longitudinal_relaxation=[];
    l_stiffness=[];
end
% waitbar(1,hwaitbar,['读取完成' num2str(100) '%'])
end


if isempty(f_v_stiffness)==0
    file_v_stiffness_fullname=[path_effective '\' f_v_stiffness.name];
    [status,sheets,xlFormat]= xlsfinfo(file_v_stiffness_fullname);
    v_stiffness=xlsread(file_v_stiffness_fullname,'Sheet1','e23');
    t=isnan(v_stiffness);
    v_stiffness(t)=[];
else
    v_stiffness=[];
end
waitbar(1,hwaitbar,['读取完成' num2str(100) '%'])

%% 
% if isempty(f_effective_radius)==0
%     file_effective_radius_fullname=[path_effective_radius '\' f_effective_radius.name];
%     [status,sheets,xlFormat]= xlsfinfo(file_effective_radius_fullname);
%     effective_radius.vx=xlsread(file_effective_radius_fullname,'Dta0','d24:d28');
%     effective_radius.fz=xlsread(file_effective_radius_fullname,'Dta0','j24:j28');
%     effective_radius.rl=xlsread(file_effective_radius_fullname,'Dta0','n24:n28');
%     effective_radius.re=xlsread(file_effective_radius_fullname,'Dta0','o24:o28');
% else
%     effective_radius=[];
% end
% save('oooo.mat', 'combine', 'purefx', 'purefymz', 'combine1', 'purefx1', 'purefymz1','lateral_relaxtion','Longitudinal_relaxation','l_stiffness','v_stiffness' ,'effective_radius')
save('oooo.mat', 'combine', 'purefx', 'purefymz', 'combine1', 'purefx1', 'purefymz1','lateral_relaxtion','Longitudinal_relaxation','l_stiffness','v_stiffness' )

end



