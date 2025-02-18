const imgInputHelper = document.getElementById("input-img-preview");
const imgInputHelperLabel = document.getElementById("input-img-label");
const imgContainer = document.querySelector(".input-img-container");
let imgFiles = [];

//----Implement search for multiple images later on----------

// const addImgHandler = () => {
//     const file = imgInputHelper.files[0];
//     if (!file) return;
//     // Generate img preview
//     const reader = new FileReader();
//     reader.readAsDataURL(file);
//     reader.onload = () => {
//       const newImg = document.createElement("img");
//       newImg.src = reader.result;
//       if imgContainer.>1: return;
//       imgContainer.insertBefore(newImg, imgInputHelperLabel);
//     };

//     // Store img file
//     imgFiles.push(file);
//     // Reset image input
//     imgInputHelper.value = "";
//     return;
//   };
const addImgHandler = () => {
  const file = imgInputHelper.files[0];
  if (!file) return;

  // Generate img preview
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    const newImg = document.createElement("img");
    newImg.src = reader.result;

    // Check if imgContainer already contains an img element
    const existingImg = imgContainer.querySelector("img");
    if (existingImg) {
      // If there's already an img element, replace it with the new one
      imgContainer.replaceChild(newImg, existingImg);
      imgFiles[0] = file;
    } else {
      // If no img element, insert the new one before the label
      imgContainer.insertBefore(newImg, imgInputHelperLabel);
      imgFiles = [file];
    }
  };
  // Reset image input
  imgInputHelper.value = "";
};

//store the img file in an <input file element> kinda obj--FileList, (using DataTransfer) so that it can be submitted via the form
const getImgFileList = (imgFiles) => {
  const imgFilesHelper = new DataTransfer();
  imgFilesHelper.items.add(imgFiles[0]);
  return imgFilesHelper.files;
};

const customFormSubmitHandler = (ev) => {
  ev.preventDefault();//prevent form submission
  const ImgInput = document.getElementById("q-img");
  ImgInput.files = getImgFileList(imgFiles);
  ev.target.submit();// submit form to server, etc
};

imgInputHelper.addEventListener("change", addImgHandler);
document
  .querySelector(".img-input-form")
  .addEventListener("submit", customFormSubmitHandler);
