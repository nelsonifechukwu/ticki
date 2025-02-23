const imgInput = document.getElementById("input-img-preview");
const imgInputLabel = document.getElementById("input-img-label");
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
  const file = imgInput.files[0];
  if (!file) return;

  // Generate img preview
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    const newImg = document.createElement("img");
    newImg.src = reader.result; //get the input img

    // Check if imgContainer already contains an img element
    const existingImg = imgContainer.querySelector("img");
    if (existingImg) {
      // If there's already an img element, replace it with the new input
      imgContainer.replaceChild(newImg, existingImg);
      imgFiles[0] = file;
    } else {
      // If no img element, insert the new input before the label
      imgContainer.insertBefore(newImg, imgInputLabel);
      imgFiles = [file];
    }
  };
  // Reset image input
  imgInput.value = "";
};

//store the img file in an <input file element> kinda obj--FileList, (using DataTransfer) so that it can be submitted via the form
const getImgFileList = (imgFiles) => {
  const _imgFiles = new DataTransfer();
  _imgFiles.items.add(imgFiles[0]);
  return _imgFiles.files;
};

const customFormSubmitHandler = (ev) => {
  ev.preventDefault(); //prevent form submission
  const _ImgInput = document.getElementById("q-img");
  _ImgInput.files = getImgFileList(imgFiles);
  ev.target.submit();// submit form to server, etc
};

imgInput.addEventListener("change", addImgHandler);
document
  .querySelector(".img-input-form")
  .addEventListener("submit", customFormSubmitHandler);
