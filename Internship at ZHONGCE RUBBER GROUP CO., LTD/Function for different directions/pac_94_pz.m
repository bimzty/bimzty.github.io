function [re,jingdu]=pac_94_pz(x,y,P);%%%%%%%%%%%%%%%,model
% P参数 P(1)~P(20)代表公式中C1~C20 P21代表C0
Fz=evalin('base','fz');
gamma=evalin('base','gamma');
% if model==2   
%     gamma=0;%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  end
options.MaxFunEvals=21000;
options.MaxIter=4000;
options.TolFun=1e-7;
options.Display0='iter';
%没有偏移
% bnd_lower=[0 -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -5 -inf -inf -inf -0.2 0];
% bnd_upper=[3 inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf 15 inf inf inf 0.2 3];
bnd_lower=[-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  -inf -inf -inf -inf -inf -inf -2 0  -inf -inf   2];
bnd_upper=[3.5 2   20   20   2   2   2   5   inf   2   2   2   2   2   2   2   2   2   0.003   2      3];
ffx=@(P,x)mz(P,x)
[re,resnorm,residual]=lsqcurvefit(ffx,P,x,y,[bnd_lower],[bnd_upper],options);
jingdu=100*sqrt(resnorm/(sum(ffx(re,x).*ffx(re,x))));
plot(x,ffx(re,x),'r',x,y,'b' );
saveas(gcf,'mz.fig');
function mz=mz(P,x)
    %x代表侧偏角
    C=P(21);
    D=(P(1).*Fz.*Fz+P(2).*Fz).*(1-P(18).*gamma.*gamma);
    K=(P(3).*Fz.*Fz+P(4).*Fz).*(1-(P(6).*abs(gamma))).*exp(-P(5).*Fz);
    B=K./C./D;
    Sh=P(11).*Fz+P(12)+P(13).*gamma;
    Sv=P(14).*Fz+P(15)+(P(16).*Fz.*Fz+P(17).*Fz).*gamma;
    x1=x+Sh;
    E=(P(7).*Fz.*Fz+P(8).*Fz+P(9)).*(1-(P(19).*gamma+P(20)).*sign(x1))./(1-(P(10).*abs(gamma)));
    mz=D.*sin(C.*atan(B.*x1-E.*(B.*x1-atan(B.*x1))))+Sv;
end

end