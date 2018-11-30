// This macro helps you to create binary segmentation maps that are required to train "DeepFLaSH" - Preprint can be found on bioRxiv: 473199


input_ROIs = getDirectory("Please select the directory that contains the ROI files");
output = getDirectory("Please select directoy where the segmentation maps shall be saved");
File.makeDirectory(output + "/black_map");
output_black_map = output + "/black_map/";
File.makeDirectory(output + "/segmentation_maps");
output_seg_maps = output + "/segmentation_maps/";
setBackgroundColor(0, 0, 0);
setForegroundColor(255, 255, 255);
setBatchMode(true);


generate_black_map ();

list_ROI_files = getFileList(input_ROIs);
Array.sort(list_ROI_files);

for (i = 0; i < list_ROI_files.length; i++) {
	generate_seg_maps (list_ROI_files[i]);
}

showMessageWithCancel("Watershading"," You have the option to perform watershading on your segmentation maps. \n This will create a second folder so that the original masks are not lost. \n Click 'OK' if you want to perform watershading or 'Cancel' to exit the macro");

list_masks = getFileList(output_seg_maps);
Array.sort(list_masks);

File.makeDirectory(output + "/watershaded_seg_maps");
output_water = output + "/watershaded_seg_maps/";

for (i = 0; i < list_masks.length; i++) {
	watershading (list_masks[i]);
}


function generate_black_map () {
	waitForUser(" In the next step, please select a representatory \n microscopy image to extract the pixel dimensions. \n Please ensure that the image is in the tiff file format");
	open();
	height = getHeight();
	width = getWidth();
	makeRectangle(0, 0, width, height);
	run("Clear", "slice");
	saveAs("Tiff", output_black_map + "black_map");		
}

function generate_seg_maps (filename) { 
	count = IJ.pad(i+1,4);
	open(output_black_map + "black_map.tif");
	roiManager("Open", input_ROIs + filename);
	roiManager("set Fill Color", "white");
	roiManager("select all");
	roiManager("fill");
	run("RGB Color");
	saveAs("Tiff", output_seg_maps + count + "_mask");
	roiManager("select all");
	roiManager("delete");
	run("Close All");
}

function watershading (filename) {
	count = IJ.pad(i+1,4);
	open(output_seg_maps + filename);
	run("8-bit");
	run("Invert LUT");
	run("Watershed");
	run("Invert LUT");
	saveAs("Tiff", output_water + count + "_watershaded_mask");
}
