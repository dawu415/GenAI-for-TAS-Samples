# Using metrics with drain logs
Here are instructions for using metrics to filter and report draining logs from Cloud Foundry. It also describes how to detect and diagnose system problems based on dropped metrics.

## How app logs are collected
The Diego Runtime reads the apps `stdout` and `stderr`, packs the logs into a common format, and sends the data to the forwarder agents. The forwarder agents aggregate the various ways of reading the logs and metrics on the VM. The forwarder agent then forwards the data to the Syslog agent and the Loggregator agent. The Syslog agent gathers all app logs, container metrics, and platform component metrics and forwards them to various syslog drains, based on the configuration. For more info, see [loggregator-agents-release](https://github.com/cloudfoundry/loggregator-agent-release) on GitHub.

## Ingress and egress metrics
There are two ingress metrics with different values for the `drain_scope` tag:

* `agent` - shows the overall number of log envelopes ingested by the syslog agent

* `all_drains` - shows the overall number of log envelopes written to the separate drain buffers
There are two layers of buffering of the ingested messages:

* The first layer is a single buffer connected to the forwarder agent. This doesn’t involve filtering. The buffer receives everything. This buffer also has an `ingress_dropped` metric, which is incremented when a message cannot be ingested.

* The second layer is made up of multiple buffers, one per syslog drain. The EnvelopeWriter reads messages from the first layer buffer, matches the source ID (app GUID or platform component name), selects the proper drain buffer, and writes the log envelope to it. There is no ingress metric here. The messages from these buffers are read, and based on the syslog scheme configuration, they are forwarded to the external Syslog servers.
The egress metric shows the number of logs forwarded with the syslog drain. The `dropped_logs` metric shows the number of failures and where the messages were dropped. Drops can happen when, for example, the Syslog server is overloaded and cannot process any more messages. The addition of the `drain_scope` and `drain_url` tags allows you to quickly see if there are some log drops in a particular drain and what kind of drain that is. If it’s an `app` drain, it means that the third-party logging system cannot consume the messages. For `aggregate` drains, it is important that they work properly. Otherwise, you might collect platform metrics and send them to your platform monitoring, and if the platform monitoring doesn’t work properly, you don’t have a good overview of what’s happening in the system.

## Checking for dropped metrics
Egress drops could mean that you are getting the wrong picture of the current state of the system, and you might not take appropriate or adequate action when something is wrong. You are generally not very concerned if an application syslog drain has egress drops, but if the metrics in the platform monitoring are missing, that will definitely be an indication you need to take some remedial action.
If there are many ingress drops on particular VM and you see higher egress rates on an application’s drains, it might mean that some application has problems and has started to log too much, or if this happens on multiple applications, you might need to scale the Diego Cells. You can use aggregate metrics based on the `drain_scope` value to generate to check the usage and generate alerts, if needed.
There are also syslog drain egress dropped metrics. The metric that gives you the drops per drain is `messages_dropped_per_drain`. Using the `drain_url` and setting the direction to `egress`, you can filter by drain to see if there are drops. If you use the same `drain_url` for apps and aggregate drains, you can filter on the `drain_scope` tag as well. These metrics are available “out of the box.”
Alternatively, you can correlate ingress and egress messages based on the source ID (app GUID). For example, you might have egress drops on a drain with a given `drain_url`. You can read the egress dropped metric based on the value of the `drain_url` tag and get the value of the `SourceId` tag. With the `SourceId` tag, you can check the ingress for that `SourceId`. If you want to inspect it even more deeply and you have applications with multiple instances, you can analyze the `InstanceId` tag to get the app instance GUID. With this information, you can use cf API calls to get the `org`, `space`, and `org users`, for example, and take some action, like contacting the app owners or developers.
Using the `drain_scope` and `drain_url` to get information about dropped metrics:

* `drain_scope` - The aggregate drains are special drains that send everything to the specified URL, which is generally the logging and monitoring system for the whole platform (Loggregator gathers and forwards app logs, app container metrics, and platform component metrics), so it is important to see that everything is forwarded and there are no drops.

* `drain_url` - If you see problems, higher loads, or more drops on a particular syslog drain, you can identify the problem app based on the `drain_url`. Syslog drain egress drops mean that the syslog server is not properly scaled to handle the required number of messages. You can set it up so that an alert is triggered to inform the app owners (customers) to check their systems. This helps you avoid outage tickets.
Generally speaking, adding these tags allows you to get a fine-grained overview of what’s happening with the syslog drains, and then proactively take actions to avoid outages in your foundations. You can also keep your customers informed about the state of their syslog drains, and possibly overloaded servers.

### Example queries (using Grafana and InfluxDB)
Here are some example queries using Grafana and InfluxDB for monitoring.

* Aggregate drain egress drops per second on Diego-Cells:
`SELECT non_negative_derivative(mean("messages_dropped_per_drain"), 1s) FROM "CF.syslog_agent" WHERE $timeFilter AND "drain_scope" = 'aggregate' AND "instance" = 'diego-cell' GROUP BY time($__interval), "deployment", "instance", "instance_id", "drain_url" fill(null)`

* App drain egress per second on Diego-Cells:
`SELECT non_negative_derivative(mean("egress"), 1s) FROM "CF.syslog_agent" WHERE $timeFilter AND "instance" = 'diego-cell' AND "drain_scope" = 'app' GROUP BY time($__interval), "instance", "instance_id", "drain_url" fill(null)`

* For syslog agent ingress metrics on Diego-Cells:
`SELECT non_negative_derivative(mean("ingress"), 1s) FROM "CF.syslog_agent" WHERE $timeFilter AND "scope" = 'agent' AND "instance" = 'diego-cell' GROUP BY time($__interval), "instance", "instance_id" fill(null)`
Because the transfer from the first level buffer to the second level syslog drain buffers happens in the same app, the syslog agent, the metric should have the same values despite the value set for the scope “agent” or “all\_drains.” These examples use “agent” for consistency because, for agent, there is an ingress metric, and an ingress dropped metric.