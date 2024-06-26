# Get started with the Notifications Service in Cloud Foundry
You can use the Notifications Service in Cloud Foundry to create a client, obtain
a token, register notifications, create a custom template, and send notifications.
For more information about the Notifications Service, see the Notifications API [v1](https://github.com/cloudfoundry-incubator/notifications/blob/master/V1_API.md) or [v2](https://github.com/cloudfoundry-incubator/notifications/blob/master/V2_API.md) documentation.

## Prerequisites
Before you use the Notifications Service you must:

* Install [Cloud Foundry](https://github.com/cloudfoundry/cf-release) (Cloud Foundry).

* Have `admin` permissions on your Cloud Foundry instance.

* Install the [Cloud Foundry Command Line Interface (cf CLI)](https://github.com/cloudfoundry/cli)
and [User Account and Authorization Server (UAAC)](https://rubygems.org/gems/cf-uaac) command line
tools.

## Create a client and get a token
To interact with the Notifications Service, you must create UAA scopes.
To create UAA scopes:

1. Target your UAA server by running:
```
uaac target uaa.YOUR-DOMAIN
```
Where `YOUR-DOMAIN` is the domain of your UAA server URL.

2. Record the **uaa:admin:client\_secret** from your deployment manifest.

3. Authenticate and obtain an access token for the admin client from the UAA server by running:
```
uaac token client get admin -s ADMIN-CLIENT-SECRET
```
Where `ADMIN-CLIENT-SECRET` is the admin client secret.
UAAC stores the token in `~/.uaac.yml`.

4. Create a `notifications-admin` client with the required scopes by running:
```
uaac client add notifications-admin --authorized_grant_types client_credentials --authorities \
notifications.manage,notifications.write,notification_templates.write,notification_templates.read,critical_notifications.write
```

* `notifications.write`: Send a notification. For example, you can send notifications to a
user, space, or everyone in the system.

* `notifications.manage`: Update notifications and assign templates for that notification.

* (Optional) `notification_templates.write`: Create a custom template for a notification.

* (Optional) `notification_templates.read`: Check which templates are saved in the database.

5. Log in using your newly created client by running:
```
uaac token client get notifications-admin
```
Stay logged in to this client to follow the examples in
this topic.
For more information about UAA scopes, see
[User Account and Authentication (UAA) Server](https://docs.cloudfoundry.org/concepts/architecture/uaa.html).

## Register notifications

**Important**
To register notifications, you must have the
`notifications.manage` scope on the client. To set critical notifications, you must have
the `critical_notifications.write` scope.
You must register a notification before sending it. Using the token `notifications-admin` from the
previous step, the following example registers two notifications with the following properties:
```
uaac curl https://notifications.user.example.com/notifications -X PUT --data '{ "source_name": "Cloud Ops Team",
"notifications": {
"system-going-down": {"critical": true, "description": "Cloud going down" },
"system-up": { "critical": true, "description": "Cloud back up" }
}
}'
```

* `source_name` has “Cloud Ops Team” set as the description.

* `system-going-down` and `system-up` are the notifications set.

* `system-going-down` and `system-up` are made `critical`, so no users can unsubscribe from that
notification.

## Create a custom template
To view a list of templates, you must have the
`notifications_templates.read` scope. To create a custom template, you must have the
`notification_templates.write` scope.
A template is made up of a name, a subject, a text representation of the template you are sending
for mail clients that do not support HTML, and an HTML version of the template.
The system provides a default template for all notifications, but you can create a custom template
by running:
```
uaac curl https://notifications.user.example.com/templates -X POST --data \
'{"name":"site-maintenance","subject":"Maintenance: {{.Subject}}","text":"The site has gone down for maintenance. More information to follow {{.Text}}","html":"<p>The site has gone down for maintenance. More information to follow {{.HTML}}"}'
```
Variables that take the form `{{.}}` interpolate data provided in the send step before a
notification is sent. Data that you can insert into a template during the send step include
`{{.Text}}`, `{{.HTML}}`, and `{{.Subject}}`.
This `curl` command returns a unique template ID that can be used in subsequent calls to refer to
your custom template. The result looks similar to:
```
{"template-id": "E3710280-954B-4147-B7E2-AF5BF62772B5"}
```
Check all of your saved templates by running:
```
uaac curl https://notifications.user.example.com/templates -X GET
```

## Associate a custom template with a notification
In this example, the `system-going-down` notification belonging to the `notifications-admin` client
is associated with the template ID `E3710280-954B-4147-B7E2-AF5BF62772B5`. This is the template ID
of the template we created in the previous section.
Associating a template with a notification requires the `notifications.manage` scope.
```
uaac curl https://notifications.user.example.com/clients/notifications-admin/notifications/system-going-down/template \

-X PUT --data '{"template": "E3710280-954B-4147-B7E2-AF5BF62772B5"}'
```
Any notification that does not have a custom template applied, such as `system-up`, defaults to a system-provided template.

## Send a notification

**Important**
To send a critical notification, you must have the
`critical_notifications.write` scope. To send a non-critical notification, you must have
the `notifications_write` scope.
You can send a notification to the following recipients:

* A user

* A space

* An org

* All users in the system

* A UAA scope

* An email address
For more information, see
[Notifications V1 Documentation](https://github.com/cloudfoundry-incubator/notifications/blob/master/V1_API.md) in
the Notifications repository on GitHub.
The following example command sends the `system-going-down` notification described above to all
users in the system:
```
uaac curl https://notifications.user.example.com/everyone -X POST --data \
'{"kind_id":"system-going-down","text":"The system is going down while we upgrade our storage","html":"<strong>THE SYSTEM IS DOWN</strong><p>The system is going down while we upgrade our storage</p>","subject":"Upgrade to Storage","reply_to":"no-reply@example.com"}'
```