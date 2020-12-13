Relocating to Another City: 

Abstract

For several reasons, especially business, many people have to relocate. Most of the time, people know their target city (or area of cities), but have to spend lots of time on researching where exactly they should live. In other words, they need to survey which neighbourhood in the large city might fit themselves and their habits. Here, I propose a recommendation model that analyzes and clusters neighbourhoods of a target city into multiple groups with different characters; and compare them with one’s current neighbourhood, in order to recommend those in the target city that are most similar to one’s hometown. In in this project, we would apply and compare the two most common algorithms in clustering, i.e., DBSCAN and KMeans, to evaluate the feasibility of these algorithms in neighbourhoods clustering.

Introduction.

Relocation has never been an easy decision for anyone. It usually takes time for researching with unstructured information on the Internet. Thanks to advancements in data science, many models and libraries are now available for building such systems that can help people shorten and prioritize the list of neighbourhoods of interest, hence reduce the task load and increase the efficiency.

In order to illustrate the problem, we will take a specific case as an example: Andy has lived in Vancouver for 25 years of his life, currently, he graduated from his grad school and looked for a data scientist position. He finally got an offer, however, the company bases in Toronto. He decided to relocate to Toronto for something new. The question here is that where exactly he could live in that big city.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539418630903_Fig1.png)

One of the concern of Andy is that he also loves his hometown so much and really wants something similar that could help him to be less homesick. In order to determine the similarity, we will need to somehow describe each neighbourhood as a numerical vector then apply some machine learning technique, e.g., DBSCAN, KMeans, to cluster them into different groups.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539418255332_Fig2.png)

This problem is not restricted to this example but has wider applications in different situations. For example, if a restaurant decides to open another branch somewhere on the other side of the city, they might also need to find a similar neighbourhood, because there are always interactive effects between stores. For instance, Starbucks usually opens their stores nearby shopping areas or even inside shopping malls, because customers tend to drink something after hours of shopping; or 7Eleven tends to open their stores close to public areas such as train station, universities, or offices. Finding similar neighbourhood has a wide application to several situations in the real world.

Dataset and Feature Vector.

Fortunately, **FourSquare** offers free APIs for developers to access their database of venues. Each venue in their dataset is usually categorized into a venue category, which is described in their [Developers Docs](https://developer.foursquare.com/docs/resources/categories). There are 10 main categories, each includes 5-91 subcategories which explicitly describe the venue, e.g., Sushi Restaurant or Fishing Store:

| **Categories**              | **Number of subcategories** |
| --------------------------- | --------------------------- |
| Arts & Entertainment        | 36                          |
| College & University        | 23                          |
| Event                       | 12                          |
| Food                        | 91                          |
| Nightlife Spot              | 7                           |
| Outdoors & Recreation       | 62                          |
| Professional & Other Places | 41                          |
| Residence                   | 5                           |
| Shop & Service              | 145                         |
| Travel & Transport          | 34                          |
| **Total**                   | **456**                     |

In order to predict the cluster for a neighbourhood in Vancouver while the model has been trained using data of Toronto’s neighbourhoods, we have to be careful with the feature vector:

- They must have the same length (train set vs. test set).
- All component must be consistently in the same order.

Here, we use a vector of 456 components, respectively to 456 subcategories, each illustrates the number of venues in that category. Hence, we have a histogram of venue categories for every neighbourhood.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539426418823_Unknown-3.png)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539533072990_Screen+Shot+2018-10-14+at+7.48.05+AM.png)

In this example, we can see the first non-zero element is a[5] = 2, which indicates there are 2 American Restaurants in this neighbourhood (American restaurant is the 6th element in the list of 456 subcategories).

Methodology.

Before conducting any advance methods in data analysis, we need to first explore the data with simple summary of **number of venues at each neighbourhood**. This photo shows the first 3 neighbourhoods.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539533592504_Screen+Shot+2018-10-14+at+8.31.38+AM.png)

One can see, the distribution of venues over categories of each neighbourhood is slightly different from each other, e.g., “Agincourt” does not have any venue in the College & University category, “Agincourt North, L’Amoreaux East, Milliken,…” does not have neither “Arts & Entertainment” nor “College & University”.

Descriptive Statistics.

To achieve a deeper insight into the data, I also get some descriptive statistics about the different categories of venue here.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539540065320_Screen+Shot+2018-10-14+at+11.00.40+AM.png)

We can see that the range of **number of venues** are really different between neighbourhoods. For example, the third line shows descriptive statistics of the Food category, while some neighbourhood has 57 venues (the last column), some has merely 1 (the fifth to last column).

Inferential Statistics.

Next, in order to make sure that category can be a “feature” of each neighbourhood  I also conducted a oneway ANOVA to investigate the effect of category on number of venues in a neighbourhood.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539534511829_Screen+Shot+2018-10-14+at+8.52.03+AM.png)

Analysis results revealed a significant effect (***F*** **= 40.54,** ***p*** **= 2.28e-48**), which also means that we can use the **distribution of venues over categories** as a feature vector of neighbourhoods (at least in this sample, Toronto neighbourhoods).

Machine Learning.

Once we have the feature vectors of both **training** (neighbourhoods in the target city) **and testing sets** (the current hometown). We can apply several machine learning techniques in clustering to achieve your goal. KMeans and DBSCAN are two of the most popular unsupervised algorithms that we can apply to solve the current problem. However, there is no gold technique for all problems. Hence, here in this project, we tried both DBSCAN and KMeans to evaluate their feasibility in the current problem.

Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
One of the best advantage of DBSCAN is that it does not require the input of cluster numbers, as we have very little knowledge about the number of clusters for a big city like Toronto, coming up with a certain number could be very difficult at the beginning.

However, DBSCAN also has some disadvantages. First, it might be much slower than KMeans in term of time and complexity. More important, DBScan does not work well over clusters with different densities. In this project, we failed to cluster the Toronto’s neighbourhoods as DBSCAN always returns a half of the neighbourhood as outliers and the other half in a single cluster.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539535763838_DBSCAN.png)

KMeans.

Though K-Means requires the number of clusters at first, we can try with different K to see home many clusters might help us answer our question. In this project, we chose K = 10. And here is the visualization of our K-Mean clustering.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539536060901_Screen+Shot+2018-10-14+at+9.50.37+AM.png)

After the clustering model has been built, we input the feature vector of Andy’s current hometown, i.e., Vancouver downtown, to find the cluster that are most similar to the current neighbourhood.

Results.

As explained above, we only evaluate KMeans clustering result as DBSCAN failed to cluster this dataset. The result shows that Vancouver downtown is most similiar to the neighbourhoods in cluster #6 in Toronto, which includes 16 neighborhoods:

1. Chinatown, Grange Park, Kensington Market
2. Commerce Court, Victoria Hotel
3. Davisville
4. Davisville North
5. Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West
6. Design Exchange, Toronto Dominion Centre
7. East Toronto
8. Harbord, University of Toronto
9. Leaside
10. Moore Park, Summerhill East
11. Ryerson, Garden District
12. St. James Town
13. Stn A PO Boxes 25 The Esplanade
14. Studio District
15. The Beaches West, India Bazaar
16. The Danforth West, Riverdale

To identify these neighbourhood,  we visualize them again on the map.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539536607601_Screen+Shot+2018-10-14+at+9.59.51+AM.png)

Interestingly, these neighbourhoods in Toronto are also around (or close to) the **downtown area**, similarly to Andy’s current hometown which is exactly at the Vancouver downtown. To validate the result in numerical variable, we again create histograms of venues over categories for both Andy’s hometown and the neighbourhoods in cluster #6 (mean values).

![Downtown Vancouver](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539537012247_vancouver.png)

![Neighbourhoods in Cluster #6 of Toronto](https://d2mxuefqeaa7sj.cloudfront.net/s_1454ED3EC8D03710AF713D45EEDCB5FE310EE2ED7C7A5681FCBCA5FCEAF5E005_1539537046116_cluster6.png)

Honestly, I would say the histograms are really close to each other though there are still minor differences, e.g., there is no travel and transport venues in downtown Vancouver (which is not very true) while there are some in cluster #6. I believe this result is fairly useful and can be used (at least) for reference purpose.

Discussion.

Though results of unsupervised learning model are sometimes hard to be evaluated, we can always reflect them with some current knowledge. For example, if it is a recommendation system for e-commerce, we can easily compare the revenue before/after application to evaluate the system. Here, I used my geographical-social knowledge about the “downtown area” to support the validity of the results. Also, we can validate the result with several methods such as in-sample validation or descriptive statistics like the one I used above.

Another point that I would like to discuss at the end of this project is the feasibility of DBSCAN vs. KMeans in this problem (or this dataset). While DBSCAN seems to have more advantage (e.g., does not need numbers of cluster, able to detect outliers), it is not robust to clusters with different densities over high-dimension data space.

Conclusion.

Recommending likelihoods in commercial areas is an important problem, as it helps people relocating to new areas can efficiently determine their target neighbourhoods. Not only to assist personal purpose, it also benefit companies who are considering to relocate/expand their branches to other cities/areas. When considering a neighbourhood, people usually want to know what services/venues available around their residence; and whether it is similar to their current living place and/or their habits.

Thanks to the availability of FourSquare APIs, we can easily retrieve the information of interest. FourSquare also provides a nicely structured hierarchy of venue categories including 10 main categories with more than 400 subcategories. In this project, we used a histogram vector of number of venues over these 456 subcategories to describe every neighbourhood. With the data extracted of the neighbourhoods in the target city, we can apply some clustering algorithms, such as DBSCAN or KMeans, to train the model, then use that model to predict the cluster for the current/original hometown.

In this project, to illustrate the model, we used data of more than 100 neighbourhoods in Toronto as the target city to train the clustering model; and data of downtown Vancouver as the original hometown. Results suggested that only KMeans is robust for this particular problem. A KMeans model clustered Toronto’s neighbourhoods into K=10 clusters and predicted the downtown Vancouver is most likely to belong to cluster #6, which includes other 16 neighbourhoods (showed in above section).

After reflecting the result on geographical map as well as histograms of venue distribution over categories, we believe that the recommendation is fairly appropriate with likely geo-social meaning and similar distribution of venues. This model helped Andy in this example to reduce the number of potential target neighbourhoods by 85% from 103 to 16 neighbourhoods. More experiments might be needed for more solid conclusion, however, here we would conclude that KMeans might be an appropriate algorithm for clustering neighbourhoods data of venue-category histograms.