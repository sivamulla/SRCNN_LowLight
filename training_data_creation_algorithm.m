image_unfiltered_input_path = 'C:\Users\Yola\git\AdvancedReading\UnFiltered\*.jpg';
image_filtered_input_path = 'C:\Users\Yola\git\AdvancedReading\Filtered\*.jpg';
output_image_path = 'C:\Users\Yola\git\AdvancedReading\output_images\image_';

image_size = 28;

imagefiles_unfiltered = dir(image_unfiltered_input_path);
imagefiles_filtered = dir(image_filtered_input_path);
nfiles = length(imagefiles_unfiltered);    % Number of files found
for i=1:nfiles
   disp(i);
   currentfilename = imagefiles_unfiltered(i).name;
   currentfilename = strcat('UnFiltered\',currentfilename);
   currentimage = imread(currentfilename);
   images_unfiltered{i} = currentimage;
   [~,~,size_z] = size(images_unfiltered{i});
   if(size_z > 1)
     images_unfiltered{i} = rgb2gray(images_unfiltered{i});  
   end
   
   currentfilename = imagefiles_filtered(i).name;
   currentfilename = strcat('Filtered\',currentfilename);
   currentimage = imread(currentfilename);
   images_filtered{i} = currentimage;
   [~,~,size_z] = size(images_filtered{i});
   if(size_z > 1)
     images_filtered{i} = rgb2gray(images_filtered{i});  
   end
   %figure;
   %imshow(images{i});
   %figure;
   [size_y,size_x,size_z] = size(images_filtered{i});
   for j=1:250
       x_start = randi([1,(size_x-image_size)],1,1);
       y_start = randi([1,(size_y-image_size)],1,1);
       total_image = zeros(image_size,2*image_size);
       new_image_unfiltered= zeros(image_size);
       new_image_unfiltered=images_unfiltered{i}(y_start:y_start+(image_size-1), x_start:x_start+(image_size-1));
       new_image_filtered= zeros(image_size);
       new_image_filtered=images_filtered{i}(y_start:y_start+(image_size-1), x_start:x_start+(image_size-1));
       for image_inner=1:image_size
           total_image(:,image_inner)=new_image_unfiltered(:,image_inner);
           total_image(:,(image_inner)+image_size)=new_image_filtered(:,image_inner);
       end
       total_image = mat2gray(total_image);
       tmp = int2str((((i-1)*20)+j));
       image_path = strcat(output_image_path, tmp, '.png');
       imwrite(total_image, image_path);
   end
end

