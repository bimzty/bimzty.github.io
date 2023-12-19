function [re,jingdu]=pac_89_px(x,y,P)
% P参数 P(1)~P(12)代表公式中B1~B13   P14代表B0
load( 'oooo.mat','-mat','purefx1')
Fz=purefx1.fz/1000;


% Fz=evalin('base','fz');
options.MaxFunEvals=14000;
options.MaxIter=4000;
options.TolFun=1e-7;
options.Display0='iter';
% bnd_lower=[-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf 1.5];
% bnd_upper=[0  0 inf inf inf inf inf inf inf inf inf inf inf 1.7];
bnd_lower=[-inf -inf  0   -inf  -1  -1   -1  -1  -1   -inf  -inf   -inf   0   1.3];
bnd_upper=[0     0    1   0     1    0   1   1    1   0     0      0     inf  1.7];

ffx=@(P,x)fx(P,x)
[re,resnorm,residual]=lsqcurvefit(ffx,P,x,y,[bnd_lower],[bnd_upper],options);
jingdu=100*sqrt(resnorm/(sum(ffx(re,x ).*ffx(re,x))));
h=figure
plot(x,ffx(re,x),'r',x,y,'b' );
saveas(gcf,'pure_fx94.fig');
close(h)
function fx=fx(P,x)
    %x代表滑移率 
    C=P(14);
    D=P(1).*Fz.*Fz+P(2).*Fz;
    K=(P(3).*Fz.*Fz+P(4).*Fz).*exp(-P(5).*Fz);
    B=K./C./D;
    Sh=P(9).*Fz+P(10);
    Sv=P(11).*Fz+P(12);
    x1=x+Sh;
    E=((P(6).*Fz+P(7)).*Fz+P(8)).*(1-P(13).*sign(x1));
    fx=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;
end

end