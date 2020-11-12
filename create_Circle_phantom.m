clear
clc

H2O = 1;
C = 1.7;
Al = 2.7;
C2H4 =4 ;
C2H5OH = 0.789;
C5H8O2 = 1.19;
Ca = 7;
CaCO3 = 2.15;

A = zeros(512);
%%%%
% 制作6个圆模型
for i = 1 : 512
    for j = 1 : 512
        if (i - 256 + 110 * cos(0 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(0 * 2 * pi / 6))^2 < 45^2
            A(i,j) = H2O;
        end
        if (i - 256 + 110 * cos(1 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(1 * 2 * pi / 6))^2 < 45^2
            A(i,j) = C;
        end
        if (i - 256 + 110 * cos(2 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(2 * 2 * pi / 6))^2 < 45^2
            A(i,j) = Al;
        end
        if (i - 256 + 110 * cos(3 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(3 * 2 * pi / 6))^2 < 45^2
            A(i,j) = C5H8O2;
        end
        
        if (i - 256 + 110 * cos(4 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(4 * 2 * pi / 6))^2 < 45^2
            A(i,j) = C2H5OH;
        end
        
        if (i - 256 + 110 * cos(5 * 2 * pi / 6))^2 + (j - 256 + 110 * sin(5 * 2 * pi / 6))^2 < 45^2
            A(i,j) = CaCO3;
        end
    end
end

% %%%%
% % 水和钙
% for i = 1 : 512
%     for j = 1 : 512
% %         if (i - 150)^2 + (j - 256)^2 < 45^2
% %             A(i,j) = Al;
% %         end
%         if (i - 256)^2 + (j - 256)^2 < 95^2
%             A(i,j) = H2O;
%         end
%     end
% end


imshow(A,[]);
% fid = fopen('mydata.txt', 'w');
% fwrite(fid,A,'double');
% fclose(fid);
