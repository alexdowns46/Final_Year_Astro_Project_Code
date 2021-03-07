clear all

DirtyDS='C:\Users\darth\Desktop\Final Year Project\matlab_files\matlab_files\Validation data\backprojected_dataset_robust';
%DirtyDS='C:\Users\darth\Desktop\Final Year Project\matlab_files\matlab_files\Validation data\backprojected_dataset';
PristineDS='C:\Users\darth\Desktop\Final Year Project\Files\datasets\full_year_project_data';
ResultsDC(1).Rlist = ['Placeholder'];

%Index all files within dataset directory:
TestingDataset=dir([DirtyDS '\' '*.png']);

netname='TEST_FOLD';
%load(['C:\Users\darth\Desktop\Final Year Project\matlab_files\matlab_files\Networks\' networkfilename]);
netlayers=importKerasLayers('TF_UNET_RDB_E45.h5','OutputLayerType','regression','ImportWeights',true,'ImageInputSize',[512 512 1]);
%% Replace old BN layers
%placeholders=findPlaceholderLayers(netlayers);

% for i=1:numel(placeholders)
%     
%     netlayers = replaceLayer(netlayers,placeholders(i,1).Name,batchNormalizationLayer());
% 
% end
net=assembleNetwork(netlayers);

%% Test!
testinglength=15;%length(TestingDataset);

ShowImages=0;
OverallTime= tic;

for j=1:testinglength
ImageClean=append(TestingDataset(j).name); %Records the name of each file


ImageDirty=[DirtyDS '\' ImageClean];
ImageClean=[PristineDS '\' ImageClean];
ImageClean=erase(ImageClean,'.png');

cleanImage=fitsread(ImageClean);
cleanImage=im2single(cleanImage);

DirtyImage=imread(ImageDirty);
DirtyImage=im2single(DirtyImage);


OutputImage=predict(net,DirtyImage);
cleanImage=cleanImage-min(cleanImage(:)); %Normalize inputs
cleanImage=cleanImage/max(cleanImage(:));

DirtyImage=DirtyImage-min(DirtyImage(:)); %Normalize inputs
DirtyImage=DirtyImage/max(DirtyImage(:));
tic
OutputImage=predict(net,DirtyImage);

OutputImage=OutputImage/max(OutputImage(:));
OutputImage=im2single(OutputImage);

Elapsed=toc;
percentcomplete=(j*100)/testinglength;

psnr1=psnr(cleanImage(:),OutputImage(:));
rsnr1 = 20*log10(norm(cleanImage(:))/norm(cleanImage(:)-OutputImage(:))); %Compute RSNR
ssim1= ssim(real(OutputImage),real(cleanImage));
rsnr2 = 20*log10(norm(cleanImage(:))/norm(cleanImage(:)-DirtyImage(:))); %Compute RSNR
psnr2=psnr(cleanImage(:),DirtyImage(:));
ssim2= ssim(real(DirtyImage),real(cleanImage));

PSNRResults(j)=psnr1;
RSNRResults(j)=rsnr1;
SSIMResults(j)=ssim1;
TimeResults(j)=Elapsed;


Results=[" File: " TestingDataset(j).name " | RSNR: " num2str(rsnr1) " | PSNR: " num2str(psnr1) " | SSIM: " num2str(ssim1) " | Current File Time: " num2str(Elapsed) " | Average SNR: " num2str(mean(RSNRResults))]; %Store results in an array
PrReslt=[Results " | Progress: " num2str(percentcomplete) "%" " | Overall Time: " num2str(toc(OverallTime))];

ResultsDC(j).Rlist = Results;

close
clc
fprintf('%s',PrReslt);

if ShowImages==0
    figure('Name',['Image: ' TestingDataset(j).name], 'Position', [0 0 1900 1068],'visible','off')
end
if ShowImages==1
    figure('Name',['Image: ' TestingDataset(j).name], 'Position', [0 0 1900 1068],'visible','on')
end
subplot(2,2,1)
imagesc(cleanImage), axis image, colorbar, colormap gray
title('Groundtruth image');

subplot(2,2,2)
imagesc(DirtyImage), axis image, colorbar, colormap gray
title(['Dirty image. SNR: ' num2str(rsnr2) 'dB. PSNR: ' num2str(psnr2) 'dB.']);

subplot(2,2,3)
imagesc(OutputImage), axis image, colorbar, colormap gray
title(['Output image. SNR: ' num2str(rsnr1) 'dB. PSNR: ' num2str(psnr1) 'dB.']);


% subplot(2,2,4)
% imshowpair(OutputImage,cleanImage,'diff');
% title('Difference between original and network output images');
saveas(gcf,['C:\Users\darth\Desktop\Final Year Project\matlab_files\matlab_files\Network Reconstructions\' netname '\Reconstructed Figures\' TestingDataset(j).name '_SNR_' num2str(rsnr1) '_PSNR_' num2str(psnr1) '_SSIM_' num2str(ssim1) '.png']);


drawnow;
imwrite(OutputImage,['C:\Users\darth\Desktop\Final Year Project\matlab_files\matlab_files\Network Reconstructions\' netname '\Reconstructed Images\' TestingDataset(j).name '_SNR_' num2str(rsnr1) '_PSNR_' num2str(psnr1) '_SSIM_' num2str(ssim1) '.png']);

end
%PSNR Computations:
meanPSNR=mean(PSNRResults); %Find mean of RSNR
stdPSNR=std(PSNRResults); %Find std dev. of RSNR

%RSNR Computations:
meanRSNR=mean(RSNRResults); %Find mean of RSNR
stdRSNR=std(RSNRResults); %Find std dev. of RSNR

%SSIM Computations:
meanSSIM=mean(SSIMResults); %Find mean of SSIM
stdSSIM=std(SSIMResults); %Find std dev. of SSIM

%Time Computations:
meanTime=mean(TimeResults); %Find mean of the elapsed time
stdTime=std(TimeResults); %Find std dev. of elapsed time

AveragesDC=["Mean RSNR: " num2str(meanRSNR) " | RSNR Std. Dev.: " num2str(stdRSNR) "Mean PSNR: " num2str(meanPSNR) " | PSNR Std. Dev.: " num2str(stdPSNR) " | Mean SSIM: " num2str(meanSSIM) " | SSIM Std. Dev.: "  num2str(stdSSIM)," | Mean Time: " num2str(meanTime) " | Time Std. Dev.: " num2str(stdTime) " | Overall Time (Entire Dataset): " num2str(toc(OverallTime))];
writematrix(AveragesDC, [netname '_ROBUST_averages.txt']); %Save results

for j=1:100
    RARR{j}=ResultsDC(j).Rlist;
end
writecell(RARR, [netname '_ROBUST_results.txt']); %Save results


