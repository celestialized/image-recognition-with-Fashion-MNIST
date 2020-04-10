# An ROV practical crossover
We may start to realize a head and headless diagnostic mechanism for addressing PTSD from various device kernel constructs across the overlay network here. ZeroMQ and OpenMQ are on the chopping block here, with implementation details to incorporate CAN bus or other black box appropriate constructs when denoting language agnostic message forwarding through devices- including both neural networks for image processing and autopiloting as well as contextuals switching between processing elements.

## Step back
Be aware the use for Jupyter Notebooks is primarily used throughout this project as a coding IDE for Python, but has mechanised support for many languages out of the box- with a few quick config options. We may see a stripped out version of the messaging within this project hinted through things like fluent bit and messages or splunk and kubernetes (docker still worked in there). So the following drives straight to the point at dissecting notebook cross talk with documents and their  implementation details.

## Wire protocol and [Messaging](https://jupyter-client.readthedocs.io/en/latest/messaging.html#messaging) in Jupyter
Details in the Session class. Every message is serialized as a sequence of at least six blobs of bytes.
Can be a struct or other "rusty" construct for the ROV. Some pieces will need tweakingm like HMAC may just be null if its a direct port, but we don't need that wrapper behind the edge router for speed, or simpler MD5 or something for a checksum- parity bit and sum bits and value. Prefix topic for IOPub subscribers- execute_result, display_data in a pub-sub lkike model. Frontends subscribe to all topics as part of the notebook system. ROV "frontends" can be groups of tasks for any given language. Subcription would be for those tasks that need to receive certain interesting data on the network. Push data onto several streams, queues, and receive based off topic. See AWS SQS queue SNS pub-sub topics for travering past the edge location, but we may implement that via gRPC.

* We can launch jupyter notebook --no-browser

Section|Description|Use
|---|---|---|
b'u-u-i-d'|        # zmq identity(ies)|dest_addr routing prefix is 0>= socket id
b'<IDS|MSG>'|       # delimiter| tells where the socket ids quit
b'baddad42'|        # HMAC signature| might not need until at edge
b'{header}'|        # serialized header dict
b'{parent_header}'| # serialized parent header dict
b'{metadata}'|      # serialized metadata dict
b'{content}'|      # serialized content dict
b'\xf0\x9f\x90\xb1| # extra raw data buffer(s)

## Another ROV affirmation
This is purely ROV introspect here. Kernels of all different language types work on Jupyter's front end. This means with a web server-client architecture and within a browser we have communications happening much in the same way we have been describing the custom defined bridge and overlay network constructs for integration of microserves for all facets of PTSD, including production grade inlined automation for debugging and autopilot. This is a very good description for how we have been discussing the production real time diagnostics.

## JSON over ZeroMQ/WebSockets message protocol

## [ZeroMQ](http://zeromq.org/)

* 2015 Apache Kafka was left by AuthO Webtasks for log aggregation for a more stable solution.
* Pieter Hintjens gives regular workshops in ZeroMQ.
* Ian Barber spoke at the PHP UK Conference

* based off BSD sockets and opensourced
* It implements real messaging patterns like topic pub-sub, workload
distribution, and request-response. 
* language agnostic. consistent API model
* designed as a library to be linked to application
* black magic lock free messaging

They say it takes less than an hour to install and learn

### 2010 [iMatix](http://zeromq.wdfiles.com/local--files/whitepapers%3Amultithreading-magic/imatix-multithreaded-magic.pdf) whitepaper
Yet another confimation for the ROV proof of concept. We just came across this paper
which spoke on Ulf Wiger summarizing the key to Erlang's concurrency is to pass information as messages rather than shared state. In 2010 the push was geared towards being thread niave, no critical
sections, no locks, no semaphores, no race conditions. No lock convoys, no  3am nightmares about optimal granularity, no two-step dances. Escaping a 16 core threshold was not without warranted messaging decisions. Its really about perspective. Traditional message delivery to queues and RPC streams are now redefining dated socket communication constructs as a truly 2 way bidirectional option today over http2. 

Also the idea of just creating tasks that each run as one thread. This makes scaling trivial. as more instances of threads with no syncrinization necessities. this means the tasks do not block. Every thread runs async and never blocks. then with no locking mechanisms or states we can see full caching with native speed. Instruction reordering, compiler optimizations etc. If we can do this, what do we get? The answer is nothing less than: perfect
scaling to any number of cores, across any number of boxes. Further, no extra
cost over normal single-threaded code. 

## Notebook document layout

<img src="frontend-kernel.png">

1 Shell: 1 router socket listens to multiple incoming connecting front ends
  * requests for code execution, object information, prompts, etc
  * sequence of request/reply actions from each frontend and the kernel
2 IO Pub is another socket acting as the 'broadcast channel' 
  * kernel publishes all side effects stdout, stderr, etc., as well as the requests coming from any client over the shell socket and its own requests on the stdin socket. (python prints to sys.stdout)
  * Muticlient frontends need to be able to know what each other has sent to the kernel (this can be useful in collaborative scenarios, for example)
  * made availabe in a uniformed way? What kinda queue?
3 Stdin: router socket connected to all front ends, and allows the kernel to request input from the active frontend when raw_input() is called. The frontend that executed the code has a DEALER socket that acts as a ‘virtual keyboard’ for the kernel while this communication is happening (in black). Tags manage the distinction of information. mess_id, which_front, etc. match the appropriate kernel that the notebook document has associated for it to use.
4 Control is another channel identical to shell on another socket for priority queue.
5 Heartbeat socket is the simple bytestring messages to be sent between the frontend and the kernel to make sure they are still connected. What they can time out? So we send frivolous info? Stay alive. whatever. dont close. 

## Message format
4 dictionary structure see 00aaa. This is a usable format.
## Compatibility 
[execute](https://jupyter-client.readthedocs.io/en/latest/messaging.html#execute) and [kernel_info](https://jupyter-client.readthedocs.io/en/latest/messaging.html#msging-kernel-info) must be implemented by the kernel as well as the associated busy and idle [kernel status](https://jupyter-client.readthedocs.io/en/latest/messaging.html#status) messages.

Caveat is in the blocking nature of waiting on user input for stdin. Implement [stdin messages](https://jupyter-client.readthedocs.io/en/latest/messaging.html#stdin-messages), if not setting `allow_stdin : false` must be set in [execute requests](https://jupyter-client.readthedocs.io/en/latest/messaging.html#executen) Request may originate from the kernel, and needs frontend input. if set to true the blocking may occur.

Think of this as a headles rov diagnostic where we are in monitor only mode. Or better yet while in blocking state possibility, the interface may send as arguments simple commands to verify boundary parameters of instigate core dumps whenever a proper sequence is recognized. Might even be a target within your training dataset.
