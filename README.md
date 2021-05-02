# Shopee - Price Match Guarantee 
[Shopee - Price Match Guarantee](https://www.kaggle.com/c/shopee-product-matching/overview) challenge is to find similar images for each group.<br>

## Data
the training set metadata. Each row contains the data for a single posting. Multiple postings might have the exact same image ID, but with different titles or vice versa.
* Num. of train : 34,250
* Num. of test  : 70,000 (only the first few rows/images of the test set are published)

unzip shopee-product-matching.zip to local.

    └── train.csv
    └── test.csv
    └── sample_submission.csv
    └── train_images
        └── 55a8b996ee39086107e141014d5e651f.jpg
        └── ab8c2ccda452af172481842c1dd3b3d7.jpg
        └── ...
    └── test_images
        └── 0006c8e5462ae52167402bac1c2e916e.jpg
        └── 0007585c4d0f932859339129f709bfdc.jpg
        └── 0008377d3662e83ef44e1881af38b879.jpg    

## Feature
* posting_id - the ID code for the posting.
* image - the image id/md5sum.
* image_phash - a perceptual hash of the image.
* title - the product description for the posting.
* label_group - ID code for all postings that map to the same product. Not provided for the test set.

## Sumission
* posting_id - the ID code for the posting.
* matches - Space delimited list of all posting IDs that match this posting. Posts always self-match. Group sizes were capped at 50, so there's no need to predict more than 50 matches.

## Metric
* Metric will be evaluated based on their mean F1 score.
