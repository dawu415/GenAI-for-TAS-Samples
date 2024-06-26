# Streaming app logs to Splunk
You can follow the steps here to integrate Cloud Foundry with Splunk Enterprise for logging.

## Step 1. Create a Cloud Foundry syslog drain for Splunk
In Cloud Foundry, create a syslog drain user-provided service instance as
described in [Using third-party log management services](https://docs.cloudfoundry.org/devguide/services/log-management.html).
Choose one or more apps whose logs you want to drain to Splunk
through the service.
Bind each app to the service instance and restart the app.
Note the GUID for each app, the IP address of the Loggregator host, and the
port number for the service.
Locate the port number in the syslog URL.
For example:
`syslog://logs.example.com:1234`

## Step 2. Prepare Splunk for Cloud Foundry
For detailed information about the following tasks, see the [Splunk documentation](http://docs.splunk.com/Documentation/Splunk).

### Install the RFC5424 syslog technology add-on
The Cloud Foundry Loggregator component formats logs according to the syslog
protocol defined in [RFC 5424](http://tools.ietf.org/html/rfc5424).
Splunk does not parse log fields according to this protocol.
To allow Splunk to correctly parse RFC 5424 log fields, install the Splunk
[RFC5424 Syslog technical add-on](http://apps.splunk.com/app/978/).

### Patch the RFC5424 syslog technology add-on

1. SSH into the Splunk VM

2. Replace `/opt/splunk/etc/apps/rfc5424/default/transforms.conf` with a new
`transforms.conf` file that consists of the following text:
```
[rfc5424_host]
DEST_KEY = MetaData:Host
REGEX = <\d+>\d{1}\s{1}\S+\s{1}(\S+)
FORMAT = host::$1
[rfc5424_header]
REGEX = <(\d+)>\d{1}\s{1}\S+\s{1}\S+\s{1}(\S+)\s{1}(\S+)\s{1}(\S+)
FORMAT = prival::$1 appname::$2 procid::$3 msgid::$4
MV_ADD = true
```

3. Restart Splunk

### Create a TCP syslog data input
Create a TCP syslog data input in Splunk, with the following settings:

* **TCP port** is the port number you assigned to your log drain service

* **Set sourcetype** is `Manual`

* **Source type** is `rfc5424_syslog` (type this value into text field)

* **Index** is the index you created for your log drain service
Your Cloud Foundry syslog drain service is now integrated with Splunk.

## Step 3. Verify that the integration was successful
Use Splunk to run a query of the form:
```
sourcetype=rfc5424_syslog index=-THE-INDEX-YOU-CREATED appname=APP-GUID
```
To view logs from all apps at once, you can omit the `appname` field.
Verify that results rows contain the three Cloud Foundry-specific fields:

* **appname**: The GUID for the Cloud Foundry app

* **host**: The IP address of the Loggregator host

* **procid**: The Cloud Foundry component emitting the log
If the Cloud Foundry-specific fields appear in the log search results,
integration is successful.
If logs from an app are missing, ensure that:

* The app is bound to the service and was restarted after binding

* The service port number matches the TCP port number in Splunk