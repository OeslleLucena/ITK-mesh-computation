#include <iostream>
#include <vector>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMetaImageIO.h"
#include "itkImageRegionIterator.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryMask3DMeshSource.h"
#include "itkMeshFileWriter.h"

#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaskImageFilter.h"

int main(int argc, char * argv[])
{

  // naming output files
  std::string imageFilename = ".../NetOutput_320x320x48.raw";
  std::string ccFilename, meshFilename, morphFilename, imageFilenameRoot, \
   binImageFilename, binImageFillFilename, binImageFillErodedFilename;
  imageFilenameRoot = imageFilename.substr(0,imageFilename.find_last_of("."));
  ccFilename = imageFilenameRoot + "_cc.mhd";
  binImageFilename = imageFilenameRoot + "_bin.mhd";
  binImageFillFilename = imageFilenameRoot + "_bin_fill.mhd";
  binImageFillErodedFilename = imageFilenameRoot + "_bin_fill_ero.mhd";
  morphFilename = imageFilenameRoot + "_cleaned.mhd";
  meshFilename = imageFilenameRoot + "_mesh.vtk";

  // read raw segmented data 	
	const int width = 320;
	const int height = 320;
	const int depth = 48;
	float isovalue = 1.0; 
  int radiusSize = 3;
  int maximumNumberOfIterations = 2;
  int backgroundValue = 0;
  int foregroundValue = 1;
  int maxLabel = 3;
  int majorityThreshold = 1;

	std::ifstream fin(imageFilename, 
					  std::ifstream::in | std::ifstream::binary);
	std::vector<float> data(width * height * depth);
	std::vector<std::vector<std::vector<float>>> field(width, 
	std::vector<std::vector<float>>(height, std::vector<float>(depth)));

	if (fin)
	{
		if (!fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float)))
			std::cout << "Read failed!" << "\n";
	}
  
  // templates for functions
  typedef itk::Image<short, 3> TImage; // Input short Image
  typedef itk::ImageFileReader <TImage> TImageReader;// Reader short Images
  typedef itk::ImageFileWriter< TImage > TWriter;
  typedef itk::BinaryBallStructuringElement<TImage::PixelType, TImage::ImageDimension> \
        TStructuringElement;
  typedef itk::BinaryErodeImageFilter<TImage, TImage, TStructuringElement> \
        TBinaryErodeImageFilter;
  typedef itk::BinaryThresholdImageFilter<TImage, TImage> TBinaryThresholdImageFilter;
  typedef itk::Mesh<double, 3> TMesh;
  typedef itk::BinaryMask3DMeshSource<TImage, TMesh> TBinaryMask3DMeshSource;
  typedef itk::MeshFileWriter<TMesh> TMeshFileWriter;
  typedef itk::VotingBinaryIterativeHoleFillingImageFilter<TImage> THoleFillingImageFilter;
  typedef itk::ConnectedComponentImageFilter<TImage, TImage> TConnectedComponentImageFilterType;
  typedef itk::LabelShapeKeepNObjectsImageFilter<TImage> TLabelShapeKeepNObjectsImageFilterType;
  typedef itk::RescaleIntensityImageFilter<TImage, TImage> TRescaleIntensityImageFilter;
  typedef itk::MaskImageFilter<TImage, TImage> TMaskImageFilter;


  // creating pointers 
  TImage::Pointer inputImage = TImage::New();
  TImage::Pointer outputImage = TImage::New();
  TImage::SizeType imageSize = {{width, height, depth}};
  TWriter::Pointer writer = TWriter::New();
  // TImageReader::Pointer reader = TImageReader::New();
  TBinaryErodeImageFilter::Pointer \
        erodeFilter = TBinaryErodeImageFilter::New();
  TBinaryThresholdImageFilter::Pointer \
        binaryFilter = TBinaryThresholdImageFilter::New();
  TBinaryMask3DMeshSource::Pointer meshFilter = TBinaryMask3DMeshSource::New();
  TMeshFileWriter::Pointer meshWriter = TMeshFileWriter::New();
  THoleFillingImageFilter::Pointer holeFillFilter = THoleFillingImageFilter::New();
  TConnectedComponentImageFilterType::Pointer connectedFilter = TConnectedComponentImageFilterType::New();
  TLabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter =
    TLabelShapeKeepNObjectsImageFilterType::New();
  TRescaleIntensityImageFilter::Pointer rescaleFilter = TRescaleIntensityImageFilter::New();
  TMaskImageFilter::Pointer maskFilter = TMaskImageFilter::New();

  std::cout << "[INFO] Read raw image" << std::endl;
  // allocating image in memory
  inputImage->SetRegions(imageSize);
  inputImage->Allocate();

  // set values in the image
  itk::ImageRegionIterator<TImage> iterator(inputImage, inputImage->GetRequestedRegion());
  int idx = 0;
  iterator.GoToBegin();

  while(!iterator.IsAtEnd())
  {
    iterator.Set(data[idx]);
    ++idx;  
    ++iterator;
  }

  // extract largest connected component
  std::cout << "[INFO] Extract largest connected component" << std::endl;
  connectedFilter->SetInput(inputImage);
  connectedFilter->Update();
  
  labelShapeKeepNObjectsImageFilter->SetInput(connectedFilter->GetOutput());
  labelShapeKeepNObjectsImageFilter->SetBackgroundValue(0);
  labelShapeKeepNObjectsImageFilter->SetNumberOfObjects(1);
  labelShapeKeepNObjectsImageFilter->SetAttribute(
    TLabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);

  rescaleFilter->SetInput(labelShapeKeepNObjectsImageFilter->GetOutput());
  rescaleFilter->SetOutputMinimum(0);
  rescaleFilter->SetOutputMaximum(itk::NumericTraits<short>::max());

  maskFilter->SetInput(inputImage);
  maskFilter->SetMaskImage(rescaleFilter->GetOutput());

  writer->SetInput(maskFilter->GetOutput()); 
  writer->SetFileName(ccFilename);
  writer->Update();

  // threshold image
  std::cout << "[INFO] Threshold image" << std::endl;
  short lowerThreshold = static_cast<short>(foregroundValue);
  short upperThreshold = static_cast<short>(maxLabel);
  short outsideValue = static_cast<short>(backgroundValue);
  short insideValue = static_cast<short>(foregroundValue);

  binaryFilter->SetInput(maskFilter->GetOutput());
  binaryFilter->SetLowerThreshold(lowerThreshold);
  binaryFilter->SetUpperThreshold(upperThreshold);
  binaryFilter->SetOutsideValue(outsideValue);
  binaryFilter->SetInsideValue(insideValue);

  // write binaryFilter
  writer->SetInput(binaryFilter->GetOutput()); 
  writer->SetFileName(binImageFilename);
  writer->Update();
  
  std::cout << "[INFO] Hole filling and erosion" << std::endl;
  // creating a ball structure 
  TStructuringElement structuringElement;
  structuringElement.SetRadius(radiusSize);
  structuringElement.CreateStructuringElement();

  // hole filling 
  THoleFillingImageFilter::InputSizeType radius;
  radius.Fill(radiusSize);
  holeFillFilter->SetInput(binaryFilter->GetOutput());
  holeFillFilter->SetRadius(radius);
  holeFillFilter->SetMajorityThreshold(majorityThreshold);
  holeFillFilter->SetBackgroundValue(backgroundValue);
  holeFillFilter->SetForegroundValue(foregroundValue);
  holeFillFilter->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  // write binary segmentation holeFillFilter
  writer->SetInput(holeFillFilter->GetOutput()); 
  writer->SetFileName(binImageFillFilename);
  writer->Update();

  // erosion (agressive exclusion)
  erodeFilter->SetInput(holeFillFilter->GetOutput());
  erodeFilter->SetKernel(structuringElement);
  erodeFilter->SetForegroundValue(foregroundValue); 
  erodeFilter->SetBackgroundValue(backgroundValue); 
  erodeFilter->Update();

  // write binary eroded holeFillFilter
  writer->SetInput(erodeFilter->GetOutput()); 
  writer->SetFileName(binImageFillErodedFilename);
  writer->Update();

  // subtract structures images from cleaned version 
  TImage::Pointer erodedImage = erodeFilter->GetOutput();
  itk::ImageRegionIterator<TImage> iterator2(erodedImage, erodedImage->GetLargestPossibleRegion());
  iterator.GoToBegin();
  iterator2.GoToBegin();
  while(!iterator.IsAtEnd())
  {
    if (iterator.Get() != isovalue)
    {
      iterator2.Set(backgroundValue);
    } 
    ++iterator2;  
    ++iterator;
  }

  // write binary segmentation
  writer->SetInput(erodedImage); 
  writer->SetFileName(morphFilename);
  writer->Update();

  std::cout << "[INFO] Mesh from binary segmentation" << std::endl;
  // compute mesh from binary image
  meshFilter->SetInput(erodedImage);
  meshFilter->SetObjectValue(foregroundValue);
  meshFilter->Update();

  // write mesh 
  meshWriter->SetFileName(meshFilename);
  meshWriter->SetInput(meshFilter->GetOutput());
  meshWriter->Update();


  return 0;
}





 