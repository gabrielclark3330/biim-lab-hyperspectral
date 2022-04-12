cube_directory = 'C:\Users\shang\Desktop\HYSPIM_Desktop_20210731\data\';
cube_filename = 'test1.cube';

% find cube filename
cube_pathname = strcat(cube_directory, cube_filename);

% This load arr is a 2D array, therefore we need to convert it to the original array shape.reshaping to get original matrice with original shape.`
lines = readlines(cube_pathname, 'LineEnding', '\n');

% read cube
cube_arr = zeros(length(lines),[]);

for ii = 1:length(lines)
   line = lines(ii);
   nums = str2num(line);
   for jj = 1:length(nums)
      cube_arr(ii, jj) = nums(jj); 
   end
end
size(cube_arr)
dimA = size(cube_arr,1);               %550
dimB = int16(size(cube_arr,2)/200);    %roiY after interpolation of rotation (80-rotation)
dimC = 200;                            %len scan

cube_arr_3d = reshape(cube_arr, [dimA, dimB, dimC]);

cube = hypercube('');

