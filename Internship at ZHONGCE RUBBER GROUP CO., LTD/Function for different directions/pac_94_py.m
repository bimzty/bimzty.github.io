function [re,fit_error]=pac_94_py(x,y,P);%%%%%%%%%%%%%%,model
% P参数 P(1)~P(17)代表公式中A1~A17 P18代表A0
Fz=evalin('base','fz');
gamma=evalin('base','gamma');
%  if model==2   
%     gamma=0;%%%%%%%%%%%%%%
%  end
options.MaxFunEvals=18000;
options.MaxIter=4000;
options.TolFun=1e-7;
options.Display0='iter';
bnd_lower=[ -80 -1500 -2500 -15 -1 -0.5 -4 -0.2 -1 -0.2 -50 -150 -5 -20 -1 -20 -75 0];%原
bnd_upper=[ 0 -500 -500 -1 1 0.5 2 0.2 1 0 100 100 5 40 1 1 5 2];%原

% bnd_lower=[ -80 -1800 -2800 -15 -1 -0.5 -4 -0.2 -1 -0.2 -50 -150 -5 -20 -1 -20 -75 0];
% bnd_upper=[ 0 -1000 -1500 -1 1 0.5 2 0.2 1 0 100 100 5 40 1 1 5 2];

% bnd_lower=[ -100 -1800 -2800 -25 -5 -2 -8 -0.2 -1 -0.2 -80 -250 -10 -30 -1 -30 -100 0];
% bnd_upper=[ 0 -1500 -1500 -10 10 2 2 2 2 0 200 200 25 80 2 2 15 5];

ffx=@(P,x)fy(P,x)
[re,resnorm,residual]=lsqcurvefit(ffx,P,x,y,[bnd_lower],[bnd_upper],options);
fit_error=100*sqrt(resnorm/(sum(ffx(re,x).*ffx(re,x))));
% plot(x1,ffx(re,x),'r',x,y,'b' );
scatter(x1,ffx(re,x),5);%x1→x???
hold on%
scatter(x,y,10,'r' );%
saveas(gcf,'fy.fig');
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