# Cloud Foundry Java Client Library
You can use the Cloud Foundry Java Client Library to manage an account on a Cloud Foundry instance.
Cloud Foundry Java Client Library v1.1.x works with apps using Spring v4.x, and
Cloud Foundry Java Client Library v1.0.x work with apps using Spring v3.x. Both versions are available in the [Cloud Foundry Java Client Library](https://github.com/cloudfoundry/cf-java-client) repository on GitHub.

## Adding the Java Client Library
To obtain the correct components, see the [Cloud Foundry Java Client Library](https://github.com/cloudfoundry/cf-java-client) repository on GitHub.
Most projects need two dependencies: the Operations API and an implementation of the Client API.
For more information about how to add the Cloud Foundry Java Client Library as dependencies to a Maven or Gradle project, see the sections below.

### Maven
Add the `cloudfoundry-client-reactor` dependency (formerly known as `cloudfoundry-client-spring`) to your `pom.xml` as follows:
```
<dependencies>
<dependency>
<groupId>org.cloudfoundry</groupId>
<artifactId>cloudfoundry-client-reactor</artifactId>
<version>2.0.0.BUILD-SNAPSHOT</version>
</dependency>
<dependency>
<groupId>org.cloudfoundry</groupId>
<artifactId>cloudfoundry-operations</artifactId>
<version>2.0.0.BUILD-SNAPSHOT</version>
</dependency>
<dependency>
<groupId>io.projectreactor</groupId>
<artifactId>reactor-core</artifactId>
<version>2.5.0.BUILD-SNAPSHOT</version>
</dependency>
<dependency>
<groupId>io.projectreactor</groupId>
<artifactId>reactor-netty</artifactId>
<version>2.5.0.BUILD-SNAPSHOT</version>
</dependency>
...
</dependencies>
```
The artifacts can be found in the Spring release and snapshot repositories:
```
<repositories>
<repository>
<id>spring-releases</id>
<name>Spring Releases</name>
<url>http://repo.spring.io/release</url>
</repository>
...
</repositories>
```
```
<repositories>
<repository>
<id>spring-snapshots</id>
<name>Spring Snapshots</name>
<url>http://repo.spring.io/snapshot</url>
<snapshots>
<enabled>true</enabled>
</snapshots>
</repository>
...
</repositories>
```

### Gradle
Add the `cloudfoundry-client-reactor` dependency to your `build.gradle` file as follows:
```
dependencies {
compile 'org.cloudfoundry:cloudfoundry-client-reactor:2.0.0.BUILD-SNAPSHOT'
compile 'org.cloudfoundry:cloudfoundry-operations:2.0.0.BUILD-SNAPSHOT'
compile 'io.projectreactor:reactor-core:2.5.0.BUILD-SNAPSHOT'
compile 'io.projectreactor:reactor-netty:2.5.0.BUILD-SNAPSHOT'
...
}
```
The artifacts can be found in the Spring release and snapshot repositories:
```
repositories {
maven { url 'http://repo.spring.io/release' }
...
}
```
```
repositories {
maven { url 'http://repo.spring.io/snapshot' }
...
}
```

## Sample code
The following is a very simple sample app that connects to a Cloud Foundry instance, logs in, and displays some information about the Cloud Foundry account. When running the program, provide the Cloud Foundry target API endpoint, along with a valid user name and password as command-line parameters.
```
import org.cloudfoundry.client.lib.CloudCredentials;
import org.cloudfoundry.client.lib.CloudFoundryClient;
import org.cloudfoundry.client.lib.domain.CloudApplication;
import org.cloudfoundry.client.lib.domain.CloudService;
import org.cloudfoundry.client.lib.domain.CloudSpace;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
public final class JavaSample {
public static void main(String[] args) {
String target = args[0];
String user = args[1];
String password = args[2];
CloudCredentials credentials = new CloudCredentials(user, password);
CloudFoundryClient client = new CloudFoundryClient(credentials, getTargetURL(target));
client.login();
System.out.printf("%nSpaces:%n");
for (CloudSpace space : client.getSpaces()) {
System.out.printf(" %s\t(%s)%n", space.getName(), space.getOrganization().getName());
}
System.out.printf("%nApplications:%n");
for (CloudApplication application : client.getApplications()) {
System.out.printf(" %s%n", application.getName());
}
System.out.printf("%nServices%n");
for (CloudService service : client.getServices()) {
System.out.printf(" %s\t(%s)%n", service.getName(), service.getLabel());
}
}
private static URL getTargetURL(String target) {
try {
return URI.create(target).toURL();
} catch (MalformedURLException e) {
throw new RuntimeException("The target URL is not valid: " + e.getMessage());
}
}
}
```
For more details about the Cloud Foundry Java Client Library, see the
[Cloud Foundry Java Client Library](https://github.com/cloudfoundry/cf-java-client) repository on GitHub.
To view the
objects that you can query and inspect, see [domain package](https://github.com/cloudfoundry/cf-java-client/tree/main/cloudfoundry-client/src/main/java/org/cloudfoundry/client/v2/domains) in the cloudfoundry/cf-java-client repository on GitHub.