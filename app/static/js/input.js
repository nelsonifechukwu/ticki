const img_input = document.getElementById("input-img-preview");
const img_input_label = document.getElementById("input-img-label");
const img_container = document.querySelector(".input-img-container");
const input_form_element = document.querySelector(".img-input-form");
const multiple_faces_form_element = document.getElementById(
  "multiple-faces-form"
);
const preview_img_handler = () => {
  const file = img_input.files[0];
  if (!file) return;

  // Generate img preview
  const reader = new FileReader();
  reader.readAsDataURL(file);

  reader.onload = () => {
    const new_img = document.createElement("img");
    new_img.src = reader.result; //get the input img

    // Check if imgContainer already contains an img element
    const existing_img = img_container.querySelector("img");
    if (existing_img) {
      img_container.replaceChild(new_img, existing_img);
    } else {
      img_container.insertBefore(new_img, img_input_label);
    }
  };
};

const custom_form_submit_handler = async (ev) => {
  ev.preventDefault();

  const q_img_input = document.getElementById("q-img");

  //check if there's any input img
  if (!img_input.files || img_input.files.length === 0) {
    const required_content = document.querySelector(".is-required");

    required_content.classList.remove("is-required");
    void required_content.offsetWidth; // force reflow
    required_content.classList.add("is-required");
    required_content.style.display = "block";
    alert("Input image is required");

    return;
  }

  q_img_input.files = img_input.files;
  form = ev.target; // don't submit form to server w/.submit()

  // Prepare form data
  const form_data = new FormData(form);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: form_data,
    });

    const html_text = await response.text();
    const parsed_doc = new DOMParser().parseFromString(html_text, "text/html");

    // Find the newly rendered selected-imgs div from response
    const new_selected_imgs = parsed_doc.querySelector(".selected-imgs");
    const current_selected_imgs = document.querySelector(".selected-imgs");

    if (new_selected_imgs && current_selected_imgs) {
      current_selected_imgs.innerHTML = new_selected_imgs.innerHTML;
    } else {
      console.warn("Could not update .selected-imgs from response.");
    }

    //Find newly rendered .multiple_faces_container from response
    const current_multiple_faces_container = document.querySelector(
      ".multiple-faces-container"
    );
    const new_multiple_input_faces = parsed_doc
      .querySelector(".multiple-faces-container .multiple-faces")
      ?.querySelectorAll(".selectable-face");

    if (new_multiple_input_faces.length > 0) {
      const new_multiple_faces_container = parsed_doc.querySelector(
        ".multiple-faces-container"
      );
      current_multiple_faces_container.innerHTML =
        new_multiple_faces_container.innerHTML;
      current_multiple_faces_container.style.display = "block";

      //re-bind the event listener to the replaced form
      const multiple_faces_form = current_multiple_faces_container.querySelector(
        "#multiple-faces-form"
      );
      if (multiple_faces_form) {
        multiple_faces_form.addEventListener("submit", bindMultipleFacesSubmit);
      }
    } else {
      current_multiple_faces_container.style.display = "none";
    }
  } catch (err) {
    console.error("Failed to submit form:", err);
  }
};

const bindMultipleFacesSubmit = async (ev) => {
  ev.preventDefault();
  const form = ev.target;

  //check if none of the extracted faces were selected for submission
  const selected = form.querySelectorAll(
    'input[name="selected_faces"]:checked'
  );
  if (selected.length === 0) {
    alert("Please select at least one face before submitting.");
    return;
  }

  const form_data = new FormData(form);

  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: form_data,
    });

    const html_text = await response.text();
    const parsed_doc = new DOMParser().parseFromString(html_text, "text/html");

    // Update selected images
    const new_selected_imgs = parsed_doc.querySelector(".selected-imgs");
    const current_selected_imgs = document.querySelector(".selected-imgs");

    //don't display current_multiple_faces_container
    const current_multiple_faces_container = document.querySelector(
      ".multiple-faces-container"
    );
    current_multiple_faces_container.style.display = "none";

    if (new_selected_imgs && current_selected_imgs) {
      current_selected_imgs.innerHTML = new_selected_imgs.innerHTML;
    } else {
      console.warn("Could not update .selected-imgs from response.");
    }
  } catch (err) {
    console.error("Failed to submit multiple faces form:", err);
  }
};

img_input.addEventListener("change", preview_img_handler);
input_form_element.addEventListener("submit", custom_form_submit_handler);
document.addEventListener("DOMContentLoaded", () => {
  const multiple_faces_form_element = document.getElementById(
    "multiple-faces-form"
  );
  if (multiple_faces_form_element) {
    multiple_faces_form_element.addEventListener(
      "submit",
      bindMultipleFacesSubmit
    );
    console.log("✔️ Listener attached to #multiple-faces-form");
  } else {
    console.warn("⚠️ Form not found on DOMContentLoaded");
  }
});
