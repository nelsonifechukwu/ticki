# ticki
Given an image with a face, select all images in a repository that contain the face

# Application Exposure
- Acts like a service for photographers (esp for unconscious pictures)
- Photographers and Clients will register on the application
- Clients can search for their pictures (that's consciously or unconsciously taken) per photographer and select based on quality, price, and other necessary info
- Clients can narrow down the date to easily access current pics
- Clients can receive a notification for any image that contains their face per photographer, even if it's random
- Clients do not need to book photographers anymore. They can simply go to events and take pictures with anyone and access it on the platform later on (Maybe give a constant time 2 days after events for photographers to upload)


# Build pivots
- Insert link to G-drive repository and find pic (Check if the link is valid/ has been visited before/has new changes)
- Use a manual folder named database w/faces subdirectory containing each_face and its embeddings
- Use threading to grab all images at once 9dask, parallelization, superfast python)
- Search using multiple input [images](https://onestepcode.com/style-html-img-file-input/)
