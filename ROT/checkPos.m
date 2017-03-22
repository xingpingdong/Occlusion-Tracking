function pos = checkPos(pos,h,w)
% check out if the pos is out of image 
if pos(1)<0,    pos(1)=1; end
if pos(2)<0,    pos(2)=1; end    
if pos(1)>h,    pos(1)=h; end
if pos(2)>w,    pos(2)=w; end
end