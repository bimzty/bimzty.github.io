%% 
load( 'oooo.mat','-mat')
if isempty(purefymz1)==0
% pure_fy
% [sr alpha gamma fz fx fy mz]=read('PC IA0IA5IA-5deg.xlsx');

% P=[-12.854474 -1113.3711 -4410.4698 -12.518279 -0.0024 0.065642332 0.20865589 -0.01571797 0.058287762 -0.092761964 18.649 -186.42 1.31 -0.20845 0.002318 0.664865 0.35017 1.5535];
P=[-12.854474 -1113.3711 -4410.4698 -12.518279 -0.0024 0.065642332 0.20865589 -0.01571797 0.058287762 -0.092761964 18.649 -186.42 1.31 0 0 0 0 1.5535];

[rey1,jingdu1]=pac_89_py(purefymz1.sa,purefymz1.fy,P);
%pure_mz
P1=[3.15 -0.7133 8.7134 13.4118 -0.10375 -0.005088 -0.0372 -0.1 -0.61144 0.03618 -0.002367 0.173244 -0.01768 -0.34 -1.641869 0.413 -0.2357 0.006075 -0.42525 -0.21503 2.23];
[rez1,jingdu2]=pac_89_pz(purefymz1.sa,purefymz1.mz,P1);
else
    jingdu1=[];
    jingdu1=[];
    rey1=[];
    rez1=[];
end
%% 
%pure_fx
% [sr alpha gamma fz fx fy mz]=read('´¿×Ý»¬IA0deg.xlsx');
% fz_nominal=4850;
if isempty(purefx1)==0
 P=[-28.808998 -1401.16957 101.33759 -172.59867 -0.061757933 0.015667623 0.18554619 1 0 0 0 0 0  1.65];
% P=[0 0 0 0 0 0 0 0 0 0 0 0 0 0];
[rex1 jingdu3]=pac_89_px(purefx1.sr/100,purefx1.fx,P);
save('pure94');
else
    jingdu3=[];
end
save('result94','rey1','rez1','rex1');
save('jingdu94','jingdu1','jingdu2','jingdu3');





