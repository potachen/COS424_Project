mainPath = '/Users/sagarsetru/Documents/Princeton/cos424/finalProject/facialRecognition/CroppedYale/';
cd(mainPath);
mainPathDir = dir(mainPath);
% load txt file with absolute paths to all images
fileList1 = textread('imgFilePaths.txt','%s');

corruptedFiles = {'yaleB11_P00A+050E-40',
'yaleB11_P00A+095E+00',
'yaleB11_P00A-050E-40',
'yaleB11_P00A-110E+15',
'yaleB12_P00A+050E-40',
'yaleB12_P00A+095E+00',
'yaleB12_P00A-050E-40',
'yaleB12_P00A-110E+15',
'yaleB12_P00A-110E-20',
'yaleB13_P00A+050E-40',
'yaleB13_P00A+095E+00',
'yaleB13_P00A-050E-40',
'yaleB13_P00A-110E+15',
'yaleB15_P00A-035E+40',
'yaleB16_P00A+095E+00',
'yaleB16_P00A-010E+00',
'yaleB17_P00A-010E+00',
'yaleB18_P00A-010E+00'};


% choose directory to save all background subtracted images
saveDir = '/Users/sagarsetru/Documents/Princeton/cos424/finalProject/facialRecognition/CroppedYale/background_subtracted/';
counter = 0;
showImg = 0;
showBackgroundImg = 0;
saveAs_pgm = 1;
saveAs_png = 1;
saveAs_tiff = 0;
for i = 1:length(fileList1),
    counter = counter + 1;
    cd(mainPath);
    imgFileList = textread(fileList1{counter},'%s');
    counter2 = 0;
    backgroundImgFile = imgFileList{end};
    backgroundImg = getpgmraw(backgroundImgFile);
    if showBackgroundImg == 1,
        imshow(backgroundImg,[]);
        pause
    end
    for j = 1:length(imgFileList)-1,
        counter2 = counter2 + 1;
        imgFile = imgFileList{counter2};
        remSet = findstr('bad',imgFile);
        if ~isempty(remSet),
            continue
            %disp(imgFile)
        end
        [~,fileName]=fileparts(imgFile);
        %skip = 0;
        %if ~isempty(findstr(fileName,'.bad')),
        %    disp('skip')
        %    disp(fileName)
        %   continue
        %end
        %for k = 1:length(corruptedFiles),
        %    cFile = corruptedFiles{k};
        %    if strcmp(cFile,fileName),
        %        skip = 1;
                %cFile
        %    end
        %end
        %if skip == 1,
        %    continue
        %    disp('skip')
        %    counter
        %end
        % do background subtraction if ambient image is properly cropped
        img = getpgmraw(imgFile);
        if size(backgroundImg)==size(img),
            img = img - backgroundImg;
            printWrongSize = 0;
        else
            printWrongSize = 1;
            %disp('wrong size')
            %counters
            continue
        end
        % save image
        cd(saveDir);
        if saveAs_pgm == 1,
            newImgName = strcat(fileName,'_bgCorrected','.pgm');
            %imwrite(img,newImgName,'pgm');
            imwrite(img,newImgName);
        end
        if saveAs_png == 1,
            newImgName = strcat(fileName,'_bgCorrected','.png');
            imwrite(img,newImgName,'png');
        end
        if saveAs_tiff == 1,
            newImgName = strcat(fileName,'_bgCorrected','.tif');
            imwrite(uint8(img), newImgName,'tiff');
            %imwrite(uint8(img), newImgName);            
        end
        %    imgSize = size(img);
        %    backgroundImg2 = imresize(backgroundImg,[imgSize(1) imgSize(2)]);
        %    img2 = img - backgroundImg2;
        if showImg == 1,
            disp(imgFile)
            imshow(img,[])
            pause;
        end
    end
    %if printWrongSize == 1,
    %    [~,fName,ext]=fileparts(backgroundImgFile);
    %    disp([fName,ext])
        %disp(size(backgroundImg))
    %else
    %    [~,fName]=fileparts(backgroundImgFile);
    %    disp(fName)
    %    disp(size(backgroundImg))
    %end
end

%%
counter = 0;
for i = 12:length(fileList1),
    counter = counter + 1;
    cd(mainPath);
    counter2 = 0;
    imgFileList = textread(fileList1{counter},'%s');
        for j = 1:length(imgFileList)-1,
            counter2 = counter2 + 1;
            imgFile = imgFileList{counter2};
            remSet = findstr('bad',imgFile);
            if ~isempty(remSet),
                disp(imgFile)
            end
        end
end
