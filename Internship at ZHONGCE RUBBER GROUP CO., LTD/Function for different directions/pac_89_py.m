function [re,jingdu]=pac_89_py(x,y,P)
% P参数 P(1)~P(17)代表公式中A1~A17 P18代表A0
load( 'oooo.mat','-mat','purefymz1')
Fz=purefymz1.fz/1000;
gamma=purefymz1.ia;

options.MaxFunEvals=18000;
options.MaxIter=4000;
options.TolFun=1e-7;
options.Display0='iter';
bnd_lower=[-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  1];
bnd_upper=[0    0     0    0    1   1     1    inf  inf  inf  100  inf  2   1    1  1  1  1.6];
ffx=@(P,x)fy(P,x)
[re,resnorm,residual]=lsqcurvefit(ffx,P,x,y,[bnd_lower],[bnd_upper],options)
jingdu=100*sqrt(resnorm/(sum(ffx(re,x).*ffx(re,x))));
 h=figure;
 plot(x,ffx(re,x),'r',x,y,'b' );

% scatter(x,ffx(re,x),'r' )
% hold on
% scatter(x,y,'*')
saveas(gcf,'pure_fy94.fig');
close(h)
function fy=fy(P,x)
    %x代表侧偏角
    C=P(18);
    D=(P(1).*Fz+P(2)).*(1-P(15).*gamma.*gamma).*Fz;
    K=P(3).*sin(atan(Fz./P(4)).*2).*(1-P(5).*abs(gamma));
    B=K./C./D;
    Sh=P(8).*Fz+P(9)+P(10).*gamma;
    Sv=P(11).*Fz+P(12)+(P(13).*Fz.*Fz+P(14).*Fz).*gamma;
    x1=x+Sh;
    E=(P(6).*Fz+P(7)).*(1-(((P(16).*gamma)+P(17)).*sign(x1)));
    fy=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;
end

end