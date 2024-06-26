# Configuring Health Monitor notifications
This topic describes how to configure notifications from the Health Monitor (HM), which monitors the health of the virtual machines in a Cloud Foundry (CF) deployment.
The HM is a BOSH component that continuously monitors all BOSH-deployed virtual machines (VMs) in a deployment. Each BOSH-deployed VM produces a heartbeat every minute and sends it to the HM, along with status and lifecycle events.
Operators can configure the HM to send an alert through notification plug-ins by editing the BOSH manifest and redeploying.

## Step 1: Set up BOSH Director credentials
To enable HM notifications, you must provide the HM with credentials to access the BOSH Director. Perform the procedures below to give the HM the correct credentials.

1. Open your BOSH manifest and locate `hm: director_account` under `properties`:
```
hm:
director_account:
ca_cert: "CA-CERT"
client_id: UAA-CLIENT-ID
client_secret: UAA-CLIENT-SECRET
password: PASSWORD
user: USERNAME
```

2. To enable the HM to access the BOSH Director, do one of the following:

* Provide the `user` and `password` for a user that can access the BOSH Director, and remove the other lines under `director_account`.

* Provide a UAA client ID for `client_id`, a UAA client secret for `client_secret`, and a certificate to verify the UAA endpoint for `ca_cert`. Remove the other lines under `director_account`.

## Step 2: Configure notifications
Perform the procedures below to set the logging level of the HM and to configure how you receive notifications.

1. Locate the `loglevel` property in your BOSH manifest.
```
hm:
loglevel: info
```
This property sets the logging level of the HM. You can set `loglevel` to `fatal`, `error`, `warn`, `info`, or `debug`.

2. You can enable notifications by e-mail, PagerDuty, AWS CloudWatch, DataDog, OpenTSDB, and Graphite. Follow the instructions below for the appropriate plug-in.

### Configure email
Replace the placeholders with the values appropriate for your deployment.
```
hm:
email_notifications: true
email_recipients: RECIPIENT1@EXAMPLE.COM, RECIPIENT2@EXAMPLE.COM
smtp:
from: SENDER-ADDRESS
host: SENDER-SMTP-HOST
port: SENDER-SMTP-PORT
domain: SENDER-SMTP-DOMAIN
tls: TRUE-OR-FALSE
auth: SMTP-AUTH-TYPE
user: SMTP-USER
password: SMTP-PASSWORD
```

* `email_notifications`: Set to `true`.

* `email_recipients`: Provide a comma-delimited list of recipient addresses.

* `smtp.from`: Provide the email address of the sender of the notifications. For example, `notifications@example.com`.

* `smtp.host`: Provide the address of the SMTP server. For example, `smtp.example.com`.

* `smtp.port`: Provide the port of the SMTP server. For example, `25`, `465`, or `587`.

* `smtp.domain`: Provide the SMTP EHLO domain. This is typically the server’s FQDN, such as `cloudfoundry.example.com`.

* `tls`: Set `tls` to `true` to enable automatic STARTTLS.

* `auth`: Provide the SMTP authentication type. Only `plain` is supported.

* `user`: If you set `auth` to `plain`, provide the username for SMTP authentication.

* `password`: If you set `auth` to `plain`, provide the password for SMTP authentication.
To customize the contents of your notification email, see the [Getting started with the notifications service](https://docs.cloudfoundry.org/adminguide/notifications.html) topic.

### Configure PagerDuty
Replace the placeholders with the values appropriate for your deployment.
```
hm:
pagerduty_enabled: true
pagerduty:
service_key: YOUR-PAGERDUTY-SERVICE-KEY
http_proxy: YOUR-HTTP-PROXY
```

* `pagerduty_enabled`: Set to `true`.

* `pagerduty.service_key`: Provide the PagerDuty service API key. For more information about how to generate an API key, see the PagerDuty [documentation](https://v2.developer.pagerduty.com/docs/events-api).

* `pagerduty.http_proxy`: Optionally, provide a HTTP proxy to connect to PagerDuty.

### Configure AWS CloudWatch
Replace the placeholders with the values appropriate for your deployment.
```
hm:
cloud_watch_enabled: true
aws:
access_key_id: YOUR-AWS-ACCESS-KEY-ID
secret_access_key: YOUR-SECRET-AWS-ACCESS-KEY
```

* `cloud_watch_enabled`: Set to `true`.

* `aws.access_key_id`: Provide the access key ID for your Amazon Web Services (AWS) account. For more information about Amazon CloudWatch, see the [CloudWatch documentation](https://aws.amazon.com/documentation/cloudwatch/).

* `aws.secret_access_key`: Provide the secret access key for your AWS account.

### Configure DataDog
Replace the placeholders with the values appropriate for your deployment.
```
hm:
datadog_enabled: true
datadog:
api_key:
application_key:
pagerduty_service_name:
```

* `datadog_enabled`: Set to `true`.

* `datadog.api_key`: Provide the API key for DataDog. For more information about DataDog, see the [DataDog documentation](http://docs.datadoghq.com/api/).

* `datadog.application_key`: Provide the HM application key for DataDog.

* `datadog.pagerduty_service_name`: Provide the service name to alert in PagerDuty on HM events.