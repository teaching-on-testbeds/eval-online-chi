


::: {.cell .markdown}

## Evaluate and monitor a model online

In addition to an offline evaluation, we also need to evaluate and monitor the ML system online, when it is "live" and accepting requests on "production data".

In this section, we will practice online evaluation of a model! After you finish this section, you should understand how to:

* monitor operational metrics in a live service
* monitor model predictions in a live service
* monitor drift in a live service

(In a separate experiment, we'll practice "closing the feedback loop," and try online evaluation of models with human feedback.)

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
import os, base64, time, random, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

```
:::

::: {.cell .markdown}

### Monitor operational metrics

:::

::: {.cell .markdown}

To monitor a service using Prometheus, it needs to expose an HTTP endpoint (by convention, `/metrics`) that Prometheus can scrape. For standard frameworks like Flask or FastAPI (or Triton inference server, for that matter) it is easy to use existing libraries to quickly instrument a service. 

We have already started an [updated FastAPI service](https://github.com/teaching-on-testbeds/eval-online-chi/tree/main/fastapi_pt), which looks very similar to one we've developed previously, but now:

* in `requirements.txt`, we've added the `prometheus-fastapi-instrumentator` library as a dependency
* and inside [the app itself](https://github.com/teaching-on-testbeds/eval-online-chi/blob/main/fastapi_pt/app.py), we've added this to the beginning:

```python
from prometheus_fastapi_instrumentator import Instrumentator
```

* and this to the end:

```python
Instrumentator().instrument(app).expose(app)
```

Meanwhile, we have configured our Prometheus service to get metrics from this service every 15 seconds, in `prometheus.yml`:

```
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'food11'
    static_configs:
      - targets: ['fastapi_server:8000']
```


That's all it takes to start sharing the basic operational metrics of a FastAPI service!


:::

::: {.cell .markdown}


Now, if we visit

```
http://A.B.C.D:8000/docs
```

in a browser, but using the floating IP assigned to our instance in place of `A.B.C.D`, we'll note that in addition to a `/predict` endpoint, our FastAPI service now has a `/metrics` endpoint. This is the endpoint that will be scraped by Prometheus. 

Click on "/metrics", then "Try it out" and "Execute" to see what it returns.


Before we look at the actual metrics in Prometheus, let's generate some data from operational use. 

In a browser, we'll visit 

```
http://A.B.C.D:5000
```

but using the floating IP assigned to our instance in place of `A.B.C.D`. Then, upload a few images to the web app and submit them for classification.

Now (or, within 15 seconds!) we can look up some operational metrics related to those queries in Prometheus. 

In a browser, we'll visit 

```
http://A.B.C.D:9090
```

but using the floating IP assigned to our instance in place of `A.B.C.D`. This will open the Prometheus web UI.

First, let's find out about the services that this Prometheus instance is monitoring. From the menu, choose "Status" > "Target Health" and note the FastAPI endpoint and its status.

Next, let's look at some of the metrics data. Prometheus uses a query language called PromQL to query metrics, but we can find some metrics of interest without knowing any PromQL. 

* From the menu, choose "Query" and then "Table". 
* Then, click on the ⋮ on the right side the query bar and choose "Explore metrics" to see some of the available metrics we can query. 
* Find "http_requests_total" in the list and click on the `{+}` to insert it into the query bar, then click "Execute". 

This will show us the cumulative number of HTTP requests processed by our FastAPI service.

Prometheus shows us metrics related to all endpoints of the FastAPI service (including the "/metrics" endpoint it scrapes itself!), but we are primarily interested in the metrics associated with the "/predict" endpoint. You'll notice that one line in the table is:

```
http_requests_total{handler="/predict", instance="fastapi_server:8000", job="food11", method="POST", status="2xx"}
```

Copy this expression and paste *that* into the query bar, then click "Execute" to only show the successful (HTTP status code 200) requests for the "/predict" endpoint.

Switch from the "Table" tab to the "Graph" tab to see a visualization of the cumulative number of requests served by this endpoint. Revisit the GourmetGram app again, and upload a few more images for classification; then execute the query in Prometheus again (or, after 15 seconds!) and confirm that it is incremented accordingly.

Instead of getting the cumulative count of requests, we may prefer to see the rate of requests per second. Put

```
rate(http_requests_total{handler="/predict", instance="fastapi_server:8000", job="food11", method="POST", status="2xx"}[1m])
```

into the query bar and click "Execute" to see the rate of successful "/predict" requests per second, averaged over a 1 minute moving window. You can use the + and - buttons to adjust the range of the horizontal axis to the time interval for which you have measurements.

To see more, let's generate some more requests. We'll send some requests directly from this notebook to the FastAPI service.

:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
image_path = "test_image.jpeg"
with open(image_path, 'rb') as f:
    image_bytes = f.read()
encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
FASTAPI_URL = "http://fastapi_server:8000/predict"
payload = {"image": encoded_str}
num_requests = 100

for _ in range(num_requests):
    response = requests.post(FASTAPI_URL, json=payload)
```
:::


::: {.cell .markdown}

Now that we have generated some data that we can query, let's use Prometheus to understand how long it takes our inference endpoint to serve requests. 

Request durations are reported as histograms, which is a slightly more involved metric type. Use the query:

```
rate(http_request_duration_seconds_sum{handler="/predict", job="food11"}[1m]) / 
rate(http_request_duration_seconds_count{handler="/predict", job="food11"}[1m])
```

to see the average duration of HTTP requests averaged over a 1 minute window. Here we are simply dividing the cumulative total time spent serving requets, by the cumulative number of served requests. Prometheus returns per-second rates, and we are averaging over a 1 minute window.

Averages can be skewed, though - we can also look at the median, using the query:

```
histogram_quantile(0.5, rate(http_request_duration_seconds_bucket{handler="/predict", job="food11"}[1m]))
```

where:

* `http_request_duration_seconds_bucket` is a histogram metric, i.e. a table of request times grouped into cumulative buckets. It counts "how many requests finished in 0.1s or less", "0.2s or less", and so on.
* `rate(...[1m])` turns the raw counts into "requests per second" over a one-minute interval for each time range bucket.
* `histogram_quantile(0.5, ...)` uses those buckets to get the 50th percentile (median) response time.

Or, to get a sense of the worst-case user experience, we can check the 95th percentile using a similar query:

```
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict", job="food11"}[1m]))
```

:::


::: {.cell .markdown}


For ongoing monitoring, instead of ad-hoc queries, we will create a dashboard using Grafana.  In a browser, we'll visit 

```
http://A.B.C.D:3000
```

but using the floating IP assigned to our instance in place of `A.B.C.D`. This will open the Grafana web UI.

Sign in using the username `admin` and password `admin`. You will be prompted to change the initial password, but you can choose to "Skip" that.

Then, you'll configure it to connect to Prometheus: 

* From the menu, choose: "Connections" > "Add new connection". 
* In the "Data sources" section, search for and choose "Prometheus", then "Add new data source". 
* Set URL to http://prometheus:9090 and leave other settings at their default values.
* Click "Save & Test"

Next, we're going to build a dashboard! 

* Click "Dashboards" > "Create dashboard" and "Add visualization".
* Select your Prometheus data source.
* Near the bottom, you will see the query builder for your Prometheus data source. Click the "Code" button to toggle from "Builder" mode to "Code" mode. 
* Put the following query (which computes average request duration in milliseconds) in the box where it says "Enter a PromQL query" and then click "Run queries":

```
1000 * rate(http_request_duration_seconds_sum{handler="/predict", job="food11"}[1m]) / 
rate(http_request_duration_seconds_count{handler="/predict", job="food11"}[1m])
```

* At the top of the graph, adjust the visible time range to "Last 15 minutes", so that you can see the data
* On the right side, click the "Time series" type
* Change the panel options: Set title to "Average request duration (ms)"
* You can play with other visualization options. Try increasing "Fill opacity", set "Show points" to "never", change "Gradient mode" to "Hue", and make other visual changes. When you are satisfied with the appearance, click "Save dashboard" and give your dashboard the title "Food Classifier Service Monitoring"

Now, you can click on "Dashboards" > "Food Classifier Service Monitoring" to see your dashboard.

Let's add another panel, showing the median, 95th percentile, and 99th percentile request duration. 

* Open the dashboard, and click "Edit" in the top right. Then click "Add" > "Visualization".
* At the top of the graph, adjust the visible time range to 15 minutes, so that you will be able to see the data
* In the query editor, paste this query and "Run queries":

```
1000*histogram_quantile(0.5, rate(http_request_duration_seconds_bucket{handler="/predict"}[1m]))
```

* Then, click "Add query" and repeat for this query:

```
1000*histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict"}[1m]))
````

* Finally, click "Add query" and add the third query:

```
1000*histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{handler="/predict"}[1m]))
```
* Click on "Legend" for each query in the query builder, change it from "Auto" to "Custom", and name each series appropriately (e.g. "p50", "p95", "p99")
* On the right side, select the "Time series" option again, then change the panel options: Set title to "Request duration percentiles (ms)"
* Once again, you can play with visual options. When you are satisfied with the appearance, click "Save dashboard" to save your changes.


Return to the dashboard. You can resize panels and drag them to place them in different positions for better visibility.

:::


::: {.cell .markdown}

Repeat the process to add additional panels to your dashboard:

* "Requests per second":

```
rate(http_requests_total{handler="/predict"}[1m])
```

* and "Failed requests per second", i.e. the rate of requests that return an error status or fail (this will have "no data" for now). You'll add two queries, one for error statuses and one for failed requests:

```
# set this up as the first query
rate(http_requests_total{handler="/predict", status="4xx"}[1m])

# set this up as a second query in the same visualization
rate(http_requests_total{handler="/predict", status="5xx"}[1m])
```

Click on the legend entry to change the color of these lines to a scary-looking red.

Save and return to the dashboard. You can resize panels and drag them to place them in different positions for better visibility.

Now, let's generate variable load - we will ramp up, then down, the load on the FastAPI service. While the cell below is running, observe the effect on your Grafana dashboard (you can set the dashboard to auto refresh with a frequency of 5 seconds, to make it easier to monitor!):

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
FASTAPI_URL = "http://fastapi_server:8000/predict"
payload = {"image": encoded_str}

load_pattern = [1, 2, 3, 5, 3, 2, 1]  # number of concurrent requests in each step
delay_between_steps = 30 

def send_continuous_requests(duration_sec):
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        requests.post(FASTAPI_URL, json=payload, timeout=5)

def run_load_stage(concurrent_workers, duration_sec):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(send_continuous_requests, duration_sec) for _ in range(concurrent_workers)]
        for f in futures:
            f.result()  # Wait for all threads to finish

for load in load_pattern:
    run_load_stage(load, delay_between_steps)
```
:::

::: {.cell .markdown}

When it is done, take a screenshot of the "Food Classifier Service Monitoring" dashboard for later reference.

:::

::: {.cell .markdown}

We can also generate some error responses, to see them on our dashboard. Visit

```
http://A.B.C.D:8000/docs
```

in a browser, but using the floating IP assigned to our instance in place of `A.B.C.D`. Click on the "predict" endpoint, and "Try it out".  Set the request body to 

```
{ }
```

and click "Execute". Scroll down to see the server response, which will be 

* Code: 422
* Details: Error: Unprocessable Entity

Try again but this time, set the request body to

```
{"image": "bad data"}
```

and click "Execute". Scroll down to see the server response, which will be

* Code: 500
* Details: Error: Internal Server Error


:::

::: {.cell .markdown}

### Alert on operational metrics


Grafana and/or Prometheus can also be configured with alerting - they can email, post a message in Slack, or execute various other notification actions based on operational metrics. As a demo, let's configure Grafana to alert when there is more than 20 error or failed responses/second for at least 1 minute.

On your dashboard, click on the ⫶ in the "Failed requests" panel, and under "More", choose "New Alert Rule". 

* In "1. Enter alert rule name", name it "Failed requests"
* In "2. Define query and alert condition":
  * Change the "A" query to include both 4xx and 5xx response codes:

```
sum(rate(http_requests_total{handler="/predict", status=~"4xx|5xx"}[1m]))
```

  * Underneath the query, it should say: "Options ... Type: Instant". If it says "Type: Range", click on "Options" and change the type to "Instant".
  * Delete the existing "B" "C" and "D" expressions using the trash icon.
  * Click "Add expression" > "Threshold".  Under "Expressions" > "B Threshold", configure it to alert when input "A" (the query result) is above 20. Click "Set B as alert condition".
  * Click "Preview" to verify that no errors are raised.
* In "3. Add folder and labels", create a new folder named "Food classifier prediction service"
* In "4. Set evaluation behavior", create a new evaluation group named "Food classifier" with a 30s evaluation interval and click "Create". Set the "Pending period" to 1m.
* In "5. Configure notifications", choose to "View or create contact points". Since we have not configured this Grafana instance to actually send emails, we're going to use the "empty" contact point.
* Click "Save".

Now, if you click on "Alerting" > "Alert rules", you should see a "Food classifier" folder with a "Food classifier prediction service" group inside it, and in it, your rule on "Failed requests".

Let's generate some error responses to trigger this alert. We're executing a similar "ramp-up/ramp-down" request pattern as before, but sending either

* an empty payload, which we know will make the server return a 422 code 
* or an invalid payload, which will make the server return a 500 code

(instead of 200, which is a code indicating success).

:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
FASTAPI_URL = "http://fastapi_server:8000/predict"

load_pattern = [1, 3, 5, 10, 5, 3, 1]  # number of concurrent requests in each step
delay_between_steps = 30 

def send_continuous_requests(duration_sec):
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        payload = random.choice([{}, {"image": "bad data"}])
        requests.post(FASTAPI_URL, json=payload, timeout=5)

def run_load_stage(concurrent_workers, duration_sec):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(send_continuous_requests, duration_sec) for _ in range(concurrent_workers)]
        for f in futures:
            f.result()  # Wait for all threads to finish

for load in load_pattern:
    run_load_stage(load, delay_between_steps)
```
:::

::: {.cell .markdown}

While this runs:

* watch the dashboard! You will see visual indicator when the alert is "Pending", which means that the threshold for alerting has been met, and it will alert if this condition persists for the entire "Pending period" (which we set to 1 minute). You will also see a visual indicator when the alert is "Fired", after the condition persists for the "Pending period", and another visual indicator when the alert status returns to "Normal".
* and, on the "Alerting" > "Alert rules" page, watch the state of your alert rule and see how it cycles through these states. Although we have not configured email, we can see when an email (or other notification) *would* have been sent - when the alert is "Fired".


When it is done, take a screenshot of the "Food Classifier Service Monitoring" dashboard for later reference, with all three of the visual alert indicators visible on the dashboard.

:::

::: {.cell .markdown}


We built this dashboard with "ClickOps" but in general, we would want to keep dashboards along with the rest of our infrastructure configuration in version control. Fortunately, dashboard definitions can be exported as JSON files and loaded systematically in a Grafana container. 

From your dashboard, click "Export" > "Export as code", and download the file. 

This JSON (along with some YAML configuration files) can be used to [provision Grafana](https://grafana.com/docs/grafana/latest/administration/provisioning/), so you can just bring it up (e.g. as part of a Docker Compose or a Kubernetes service) and have it ready to go. (We won't do this now, though.)

:::


::: {.cell .markdown}

### Monitor predictions

In addition to standard "web service" metrics, like error responses and request durations, we may want to monitor application-specific metrics during operation. For example, let's monitor the confidence of the model in its predictions, and the frequency with which it predicts each class.

We'll need to make a few changes to our FastAPI service, since the auto-instrumentation of `prometheus-fastapi-instrumentator` does not include application-specific metrics.

1) Add the Prometheus Python client to the `requirements.txt` of the FastAPI container. Run

```
# runs on node-eval-online
nano eval-online-chi/fastapi_pt/requirements.txt
```

and add `prometheus-client`. Save this file with Ctrl+O and Enter, then quit `nano` with Ctrl+X.

2) Edit `app.py`: 

```
# runs on node-eval-online
nano eval-online-chi/fastapi_pt/app.py
```

Near the top, add

```python
from prometheus_client import Histogram, Counter
```

and then

```python
# Histogram for prediction confidence
confidence_histogram = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
)

# Count how often we predict each class
class_counter = Counter(
    "predicted_class_total",
    "Count of predictions per class",
    ['class_name']
)
```

Scroll to the "Run inference" section, and just before the inference result is returned, add


```python
        # Update metrics
        confidence_histogram.observe(confidence)
        class_counter.labels(class_name=classes[predicted_class]).inc()
```

to observe the confidence value, and increment the class counter for the corresponding class.

Save this file with Ctrl+O and Enter, then quit `nano` with Ctrl+X.

Re-build the container image with

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml build fastapi_server
```

and then recreate it with

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up -d
```

:::

::: {.cell .markdown}

Let's test it! In a browser, we'll visit 

```
http://A.B.C.D:5000
```

but using the floating IP assigned to our instance in place of `A.B.C.D`. Then, upload a few images to the web app and submit them for classification.

Next, open

```
http://A.B.C.D:8000/docs
```

in a browser, but using the floating IP assigned to our instance. Click on "/metrics", then "Try it out" and "Execute". Note that now, the reported metrics include `prediction_confidence_bucket` for each bucket, and `predicted_class_total` for each class.

We'll add another dashboard to Grafana to monitor the predictions. Instead of the "ClickOps" way, we'll set up this dashboard by importing it from a JSON definition.

In Grafana, click "Dashboards" > "New" > "Import" and where it says "Import via dashboard JSON model", paste:

{% raw %}
```
{
  "id": null,
  "title": "Food Classifier Prediction Monitoring",
  "timezone": "browser",
  "time": {
    "from": "now-15m",
    "to": "now"
  },
  "refresh": "5s",
  "schemaVersion": 38,
  "version": 1,
  "panels": [
    {
      "type": "timeseries",
      "title": "Average Prediction Confidence",
      "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
      "targets": [
        {
          "expr": "sum(rate(prediction_confidence_sum[1m])) / sum(rate(prediction_confidence_count[1m]))",
          "legendFormat": "avg(confidence)",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1
        }
      },
      "options": {
        "legend": {
          "displayMode": "list",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "datasource": "prometheus"
    },
    {
      "type": "timeseries",
      "title": "Prediction Confidence (Cumulative)",
      "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
      "targets": [
        {
          "expr": "rate(prediction_confidence_bucket{le!=\"+Inf\"}[1m])",
          "legendFormat": "{{le}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "ops",
          "custom": {
            "fillOpacity": 0,
            "stacking": {
              "mode": "none"
            }
          }
        },
        "overrides": []
      },
      "options": {
        "legend": {
          "displayMode": "list",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "datasource": "prometheus"
    },
    {
      "type": "bargauge",
      "title": "Prediction Confidence (Over Selected Time Range)",
      "gridPos": { "x": 0, "y": 8, "w": 12, "h": 10 },
      "targets": [
        {
          "expr": "increase(prediction_confidence_bucket{le!=\"+Inf\"}[$__range])",
          "legendFormat": "≤ {{le}}",
          "refId": "A"
        }
      ],
      "options": {
        "orientation": "horizontal",
        "displayMode": "gradient",
        "showUnfilled": true
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0
        }
      },
      "datasource": "prometheus"
    },
    {
      "type": "bargauge",
      "title": "Predicted Class Totals (Over Selected Time Range)",
      "gridPos": { "x": 12, "y": 8, "w": 12, "h": 10 },
      "targets": [
        {
          "expr": "increase(predicted_class_total[$__range])",
          "legendFormat": "{{class_name}}",
          "refId": "A"
        }
      ],
      "options": {
        "orientation": "horizontal",
        "displayMode": "gradient",
        "showUnfilled": true
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0
        }
      },
      "datasource": "prometheus"
    }
  ]
}
```
{% endraw %}

then click "Load" and "Import". Now, you have a "Food Classifier Prediction Monitoring" dashboard in addition to your "Food Classifier Service" dashboard.

Let's generate some requests to populate this dashboard with meaningful data:

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'))
image_paths = [sample[0] for sample in test_dataset.samples]
random.shuffle(image_paths)

FASTAPI_URL = "http://fastapi_server:8000/predict"

for image_path in image_paths:
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    encoded_str = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image": encoded_str}
    response = requests.post(FASTAPI_URL, json=payload)
```
:::


::: {.cell .markdown}

When it is done, take a screenshot of the "Food Classifier Prediction Monitoring" dashboard for later reference.

:::

::: {.cell .markdown}

### Alert on prediction metrics

With this dashboard, we can monitor how the predicted class distribution and prediction confidence is evolving. For example, if our model is becoming "stale" and has less confidence in its predictions, we can use that to trigger an alert and potentially, re-train the model.

We'll try it now - on the dashboard, click on the ⫶ in the "Average Prediction Confidence" panel, and under "More", choose "New Alert Rule". 

* In "1. Enter alert rule name", name it "Low prediction confidence"
* In "2. Define query and alert condition", set the Alert Condition to: When query is below 0.75
* In "3. Add folder and labels", use "Food classifier"
* In "4. Set evaluation behavior", use the evaluation group named "Food classifier". Set the "Pending period" to 1m. (In a real production setting, we would want this alert to fire only if prediction confidence is low for an extended period of time, like hours, days, or weeks, but in this example we use a small pending period so that we can see it in a shorter time frame!)
* In "5. Configure notifications", use the "empty" notification option. (In a real production service, we might configure a notification to a webhook that would trigger rebuilding, for example.)
* Click "Save".

Repeat the cell above this one to send requests for the test dataset. This will not trigger the alert, since the prediction confidence is high.

Immediately after that, run the cell below, to send prediction requests for the "cake_looks_like" samples (of photos of cake that look like other food items), where the prediction confidence will be lower:

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
image_dir = "cake_looks_like"
image_paths = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
]

FASTAPI_URL = "http://fastapi_server:8000/predict"

for _ in range(1000):
    random.shuffle(image_paths)
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        response = requests.post(FASTAPI_URL, json=payload)
```
:::

::: {.cell .markdown}

Take a screenshot of this "Food Classifier Prediction Monitoring" dashboard showing where the alert condition is triggered and where it is fired.

:::


::: {.cell .markdown}

### Monitor infrastructure metrics

Finally, we should also monitor the containers themselves. We'll use [`cAdvisor`](https://github.com/google/cadvisor).

1) First, add a `cadvisor` container to the Docker Compose YAML. Open it with

```bash
# runs on node-eval-online
nano eval-online-chi/docker/docker-compose-prometheus.yaml
```

then add this to the end of the `services` section (before the `volumes` section):

```
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    restart: unless-stopped
```

Save the file with Ctrl+O and Enter, then Ctrl+X to exit.

2) Tell Prometheus to scrape this endpoint for metrics. Open the Prometheus config with

```bash
# runs on node-eval-online
nano eval-online-chi/docker/prometheus.yml
```

and at the end, in the `scrape_configs` section, add:

```
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

Save the file with Ctrl+O and Enter, then Ctrl+X to exit.

3) Bring up the new container with

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up -d
```

and force the Prometheus container to restart with its new configuration:

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up prometheus --force-recreate -d
```

**Note**: `cAdvisor` has its own web UI. You can see it at 

```
http://A.B.C.D:8080
```

in a browser, but using the floating IP assigned to our instance in place of `A.B.C.D`. Click on "Docker containers" and then any individual container to see more details of its resource usage.


4) But, we'll prefer to monitor everything in Grafana, rather than having one system for inference service monitoring and another for infrastructure monitoring. To monitor our containers, we'll import a [pre-built Grafana dashboard for cAdvisor](https://grafana.com/grafana/dashboards/19908-docker-container-monitoring-with-prometheus-and-cadvisor/).

* In Grafana, click "Dashboards" > "New" > "Import"
* Enter this URL in the space where it says "Find and import dashboards for common applications at grafana.com/dashboards":

```
https://grafana.com/grafana/dashboards/19908-docker-container-monitoring-with-prometheus-and-cadvisor/
```

* Click "Load". 
* In the "Prometheus" section, use the drop-down menu to select your Prometheus data source. 
* Then, click "Import".

Now, you'll see the "cAdvisor Docker Insights" dashboard in the "Dashboards" section. Open the dashboard and monitor the resource usage of your Docker containers when you generate requests against the FastAPI service. (Adjust the time scale of the dashboard to 15 minutes so that you can see the data.)


:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
FASTAPI_URL = "http://fastapi_server:8000/predict"
payload = {"image": encoded_str}

load_pattern = [1, 2, 3, 5, 3, 2, 1]  # number of concurrent requests in each step
delay_between_steps = 60 

def send_continuous_requests(duration_sec):
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        requests.post(FASTAPI_URL, json=payload, timeout=5)

def run_load_stage(concurrent_workers, duration_sec):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(send_continuous_requests, duration_sec) for _ in range(concurrent_workers)]
        for f in futures:
            f.result()  # Wait for all threads to finish

for load in load_pattern:
    run_load_stage(load, delay_between_steps)
```
:::

::: {.cell .markdown}

Let this run for about ten minutes. Then, come back a few minutes later (because in this dashboard, rate metrics are over a 10 minute window) and take a screenshot of the "cAdvisor Docker Insights" dashboard for later reference. (You don't have to capture the entire dashboard, it's fine if it's just the top panels.)

:::


::: {.cell .markdown}

### Detect data drift

For machine learning systems in particular, it is important to monitor for data drift - when new production data is dissimilar to the data on which a model was trained.

There are many different tests for data drift, depending on what type of data you are working with (tabular, image, text, audio, time series). In this example, we will use the [`alibi-detect`](https://github.com/SeldonIO/alibi-detect) package, which implements many of these tests. Here's the current list of drift detection tests that are implemented:


| Detector                         | Tabular | Image | Time Series | Text  | Categorical Features | Online | Feature Level |
|:---------------------------------|  :---:  | :---: |   :---:     | :---: |   :---:              | :---:  | :---:         |
| Kolmogorov-Smirnov               | ✔       | ✔     |             | ✔     | ✔                    |        | ✔             |
| Cramér-von Mises                 | ✔       | ✔     |             |       |                      | ✔      | ✔             |
| Fisher's Exact Test              | ✔       |       |             |       | ✔                    | ✔      | ✔             |
| Maximum Mean Discrepancy (MMD)   | ✔       | ✔     |             | ✔     | ✔                    | ✔      |               |
| Learned Kernel MMD               | ✔       | ✔     |             | ✔     | ✔                    |        |               |
| Context-aware MMD                | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Least-Squares Density Difference | ✔       | ✔     |             | ✔     | ✔                    | ✔      |               |
| Chi-Squared                      | ✔       |       |             |       | ✔                    |        | ✔             |
| Mixed-type tabular data          | ✔       |       |             |       | ✔                    |        | ✔             |
| Classifier                       | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Spot-the-diff                    | ✔       | ✔     |  ✔          | ✔     | ✔                    |        | ✔             |
| Classifier Uncertainty           | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |
| Regressor Uncertainty            | ✔       | ✔     |  ✔          | ✔     | ✔                    |        |               |


We will consider a specific measure of drift called Maximum Mean Discrepenacy (MMD). This "detector" will look at the data before the final classification decision - in this case, where we have used transfer learning to fit a classification head on top of a "base" model, we'll look at the output of the "base" model. Then, MMD will compute the distance between the mean embeddings of the features from:

* the distribution of some "reference" data
* and the distribution of new samples

to see how different they are. 

Let's try this now. First, we will grab a batch of the training data to use as the "reference" data, and a sample of validation data to use as "new" data -

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
model_path = "models/food11.pth"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device, weights_only=False)
model = model.to(device)
_ = model.eval()  
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'training'), transform=val_test_transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset   = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
val_loader    = DataLoader(train_dataset, batch_size=32, shuffle=True)
```
:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
x_ref, y_ref = next(iter(train_loader))
print("x_ref shape:", x_ref.shape)
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
x_new, y_new = next(iter(val_loader))
print("x_new shape:", x_new.shape)
```
:::


::: {.cell .markdown}

Now, we will set up our drift detector.

* we will prepare a `feature_model` that gets the feature map from our model, before the classification haed
* its output will be passed to a preprocessing function 
* and then, pass the reference data  to MMD-based change detector using the reference data

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
from alibi_detect.cd.pytorch import HiddenOutput, preprocess_drift
from alibi_detect.cd import MMDDrift, MMDDriftOnline
from functools import partial
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online

# Use `HiddenOutput` from alibi_detect to extract features from the last layer of our classifier before the "head"
feature_model = HiddenOutput(model, layer=-1) 

# preprocess_drift defines the model used for drift detection - the HiddenOutput model
preprocess_fn = partial(preprocess_drift, model=feature_model, batch_size=128)

# Create an MMD-based change detector, where:
#  - x_ref is reference data to describe what input to the classifier head *should* look like
#  - preprocess_fn will be used to extract features from both reference and test data,
#  - the backend is PyTorch (Tensorflow is also supported)
#  - p_val = 0.05 means we consider the difference between reference and test data to be significant at 95% confidence
cd = MMDDrift(x_ref, backend='pytorch', p_val=.05, preprocess_fn=preprocess_fn)
```
:::

::: {.cell .markdown}

Get another batch of training data, and pass it to the detector - 

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
x_tr, y_tr = next(iter(train_loader))
cd_preds = cd.predict(x_tr)
cd_preds
```
:::

::: {.cell .markdown}

The output should show that no drift was detected (`is_drift`: `0`) because the p-value is higher than the threshold we set for statistical significance (`0.05`), and the `distance` between test and reference distributions is small -  well below the `distance_threshold`. This means that this batch of data is statistically similar to the reference data.

To use this drift detector in production, we'll need the online version of it, which accepts one sample at a time as input instead of a batch of data.

:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
preprocess_fn = partial(preprocess_drift, model=feature_model)

cd_online = MMDDriftOnline(
    x_ref,           # reference embeddings
    ert=300,         # expected run time between false positives (larger = fewer false alarms)
    window_size=10,  # how many test samples to use in each MMD check
    backend='pytorch',
    preprocess_fn=preprocess_fn
)
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
# try the online detector on evaluation data
results = []
x_new, y_new = next(iter(val_loader))  # one batch
for x in x_new:
    cd_pred = cd_online.predict(x.cpu().numpy())
    results.append(cd_pred['data'])

```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
print([r['is_drift'] for r in results])
```
:::

::: {.cell .markdown}

(ignore the first `window_size` outputs, which are not meaningful.)

:::

::: {.cell .markdown}

What if there are genuinely different images than the ones seen in training? Suppose that more GourmetGram users are submitting AI-generated images, which are very different from the images seen in training:

:::


::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
image_dir = "ai_images"
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

num_images = 16
sample_files = random.sample(image_files, min(num_images, len(image_files)))

cols = 4
rows = (len(sample_files) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

for ax, filename in zip(axes.flatten(), sample_files):
    image_path = os.path.join(image_dir, filename)
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(filename, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
# try the online detector on AI-generated image data
results_ai = []
sample_files = random.sample(image_files, min(32, len(image_files)))
for filename in sample_files:
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")
    x_tensor = val_test_transform(image)  # shape: [3, 224, 224]
    x_np = x_tensor.numpy()
    cd_pred = cd_online.predict(x_np)
    results_ai.append(cd_pred['data'])
```
:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
print([r['is_drift'] for r in results_ai])
```
:::

::: {.cell .code}
```python
# runs on node-eval-online
import matplotlib.pyplot as plt

test_stats = [r['test_stat'] for r in results[:32]]
drift_flags = [r['is_drift'] for r in results[:32]]

test_stats_ai = [r['test_stat'] for r in results_ai[:32]]
drift_flags_ai = [r['is_drift'] for r in results_ai[:32]]

combined_stats = test_stats + test_stats_ai
combined_flags = drift_flags + drift_flags_ai
combined_labels = ['regular'] * len(test_stats) + ['ai'] * len(test_stats_ai)

plt.figure(figsize=(10, 5))
for i, (stat, drift, label) in enumerate(zip(combined_stats, combined_flags, combined_labels)):
    color = 'red' if drift else 'blue'
    marker = 'o' if label == 'regular' else 's'
    plt.scatter(i, stat, color=color, marker=marker)

threshold = results[0]['threshold'] if results else results_ai[0]['threshold']
plt.axhline(y=threshold, linestyle='--', color='gray')

plt.title("Drift Detection")
plt.xlabel("Sample Index")
plt.ylabel("Test statistic")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Food11 samples', markerfacecolor='blue'),
    plt.Line2D([0], [0], marker='s', color='w', label='AI samples', markerfacecolor='red'),
    plt.Line2D([0], [0], color='gray', linestyle='--', label='threshold')
])
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown}

The drift detector should identify that the samples in the "ai_images" set are diferent from those in the Food11 validation set. (Some samples are more "like" Food11 than others - you can run this multiple times to try again with different random samples.)

:::

::: {.cell .markdown}

### Monitor for data drift

Let's integrate this into our prediction endpoint, so that drift metrics from production data will be generally available - e.g. for things like deciding when to retrain the model.

First, we need to save the change detection model:

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
from alibi_detect.saving import save_detector
cd_online.reset_state()
save_detector(cd_online, "cd")
```
:::

::: {.cell .markdown}

Now, we are ready to modify our FastAPI service!

1) Copy the saved change detector to our FastAPI service directory.

```bash
# runs on node-eval-online
cp -R eval-online-chi/workspace/cd eval-online-chi/fastapi_pt/
```

2) Open the FastAPI `app.py` to edit it:

```bash
# runs on node-eval-online
nano eval-online-chi/fastapi_pt/app.py
```

Near the top of this file, add the new import:

```python
from alibi_detect.saving import load_detector
```

and read in the change detector:

```python
# Load the change detector from file
cd = load_detector("cd")
```

then, add the new metrics that we will report to Prometheus - 

```python
# Counter for drift events
drift_event_counter = Counter(
        'drift_events_total', 
        'Total number of drift events detected'
)

# Histogram for drift test statistic
drift_stat_hist = Histogram(
        'drift_test_stat', 
        'Drift score distribution'
)
```

We are going to do drift detection asynchronously - so that the user does not have to wait for the drift detection model before getting the class label. 

Near the top, add another import:

```python
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed
```

and, right below the definition of the Prometheus metrics, add a drift detection function:

```python
def detect_drift_async(cd, x_np):
    cd_pred = cd.predict(x_np)
    test_stat = cd_pred['data']['test_stat']
    is_drift = cd_pred['data']['is_drift']

    drift_stat_hist.observe(test_stat)
    if is_drift:
        drift_event_counter.inc()
```

Then, in the predict endpoint, right before `return` - add (with the appropriate indentation level - and use spaces, not tabs, to indent)

```python
# Submit to async drift detection thread
x_np = image.squeeze(0).cpu().numpy()
executor.submit(detect_drift_async, cd, x_np)
```

Save the file with Ctrl+O and Enter, then Ctrl+X to exit.

3) Open the FastAPI `requirements.txt` to edit it:

```bash
# runs on node-eval-online
nano eval-online-chi/fastapi_pt/requirements.txt
```

and add 

```
alibi-detect
```

Save the file with Ctrl+O and Enter, then Ctrl+X to exit.

4) Re-build the container image with

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml build fastapi_server
```

and then recreate it with

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml up -d
```

Loading the FastAPI service takes a little longer, now that the change detector is part of it - use

```bash
# runs on node-eval-online
docker logs fastapi_server
```

until you see

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

:::

::: {.cell .markdown}

Let's test it! In a browser, we'll visit 

```
http://A.B.C.D:5000
```

but using the floating IP assigned to our instance in place of `A.B.C.D`. Then, upload a few images to the web app and submit them for classification.

Next, open

```
http://A.B.C.D:8000/docs
```

in a browser, but using the floating IP assigned to our instance. Click on "/metrics", then "Try it out" and "Execute". Note that now, the reported metrics include metrics related to drift.

To monitor drift live, we can add a Grafana dashboard. In Grafana, click "Dashboards" > "New" > "Import" and where it says "Import via dashboard JSON model", paste:


{% raw %}
```
{
  "id": null,
  "title": "Food Classifier Drift Monitoring",
  "timezone": "browser",
  "schemaVersion": 41,
  "version": 1,
  "refresh": "10s",
  "time": {
    "from": "now-15m",
    "to": "now"
  },
  "panels": [
      {
      "id": 1,
      "type": "stat",
      "title": "Drift Events (per second)",
      "datasource": "prometheus",
      "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
      "targets": [
        {
          "expr": "rate(drift_events_total[1m])",
          "refId": "A"
        }
      ],
      "options": {
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "orientation": "horizontal",
        "colorMode": "value",
        "graphMode": "area",         
        "textMode": "auto"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "ops",
          "decimals": 2,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "red", "value": 1 }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "id": 2,
      "type": "timeseries",
      "title": "Median Drift Score (test_stat)",
      "datasource": "prometheus",
      "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
      "targets": [
        {
          "expr": "histogram_quantile(0.5, rate(drift_test_stat_bucket[1m]))",
          "refId": "A",
          "legendFormat": "median test_stat"
        }
      ],
      "options": {
        "legend": {
          "displayMode": "list",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "fieldConfig": {
        "defaults": {
          "unit": "none",
          "decimals": 4
        },
        "overrides": []
      }
    }
  ]
}
```
{% endraw %}

Then click "Load" and "Import".

:::

::: {.cell .markdown}

To see it in action, let's generate some requests from Food11:

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'))
image_paths = [sample[0] for sample in test_dataset.samples]
random.shuffle(image_paths)

FASTAPI_URL = "http://fastapi_server:8000/predict"

for image_path in image_paths:
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    encoded_str = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image": encoded_str}
    response = requests.post(FASTAPI_URL, json=payload)
```
:::

::: {.cell .markdown}

and then some requests from the AI samples that are *not* similar to the Food11 data:

:::

::: {.cell .code}
```python
# runs on Jupyter container on node-eval-online
image_dir = "ai_images"
image_paths = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
]

FASTAPI_URL = "http://fastapi_server:8000/predict"

for _ in range(1000):
    random.shuffle(image_paths)
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        response = requests.post(FASTAPI_URL, json=payload)
```
:::

::: {.cell .markdown}

When you see that drift events are detected, take a screenshot of the "Food Classifier Drift Monitoring" dashboard for later reference.

:::


::: {.cell .markdown}

--- 

When you have finished with this section, bring down these services with 

```bash
# runs on node-eval-online
docker compose -f eval-online-chi/docker/docker-compose-prometheus.yaml down
```

:::
