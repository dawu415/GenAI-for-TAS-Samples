# Configuring service connections for Node.js apps
You can bind a data source to a Node.js
application that is deployed and running on Cloud Foundry.

## Parse VCAP\_SERVICES for credentials
You must parse the `VCAP_SERVICES` environment variable in your code to get the
required connection details such as host address, port, user name, and
password.
For example, if you are using PostgreSQL, your `VCAP_SERVICES` environment
variable might look something like this:
```
{
"mypostgres": [{
"name": "myinstance",
"credentials": {
"uri": "postgres://myusername:mypassword@host.example.com:5432/serviceinstance"
}
}]
}
```
This example JSON is simplified; yours might contain additional properties.

### Parse with cfenv
The `cfenv` package provides access to Cloud Foundry application environment
settings by parsing all the relevant environment.
The settings are returned as JavaScript objects.
`cfenv` provides reasonable defaults when running locally, as well as when
running as a Cloud Foundry application.
For more information, see the [npm website](https://www.npmjs.org/package/cfenv).

### Manual parsing
First, parse the `VCAP_SERVICES` environment variable.
For example:
```
var vcap_services = JSON.parse(process.env.VCAP_SERVICES)
```
Then pull out the credential information required to connect to your service.
Each service packages requires different information.
If you are working with Postgres, for example, you need a `uri` to
connect.
You can assign the value of the `uri` to a variable as follows:
```
var uri = vcap_services.mypostgres[0].credentials.uri
```
Once assigned, you can use your credentials as you would normally in your
program to connect to your database.

## Connecting to a service
You must include the appropriate package for the type of services your
application uses.
For example:

* Rabbit MQ through the [amqp](https://github.com/postwait/node-amqp) module

* [mongoose](http://mongoosejs.com/) modules

* MySQL through the [mysql](https://github.com/felixge/node-mysql) module

* Postgres through the [pg](https://github.com/brianc/node-postgres) module

* Redis through the [redis](https://github.com/mranney/node_redis) module

## Add dependency to package.json
Edit `package.json` and add the intended module to the `dependencies` section.
Normally, only one is necessary, but for the sake of the example, add all of them:
```
{
"name": "hello-node",
"version": "0.0.1",
"dependencies": {
"express": "\*",
"mongodb": "\*",
"mongoose": "\*",
"mysql": "\*",
"pg": "\*",
"redis": "\*",
"amqp": "\*"
},
"engines": {
"node": "0.8.x"
}
}
```
You must run `npm shrinkwrap` to regenerate your `npm-shrinkwrap.json` file
after you edit `package.json` file.