# -*- mode: python -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2016-2022  CINCH Enterprises, Ltd.  See LICENSE.txt for terms.

"""
Utilities to supplement networking stack functions.
"""
import socket
from urllib.parse import urlparse
import urllib.request


def get_ipaddr(host=None):
    """
    Retrieves the IPv4 address of the network host.

    :param host: Host to resolve (None => host unknown, retrieve by network broadcast -- must be attached)
    :type  host: Union(str, None)

    :return: IPv4 address of host
    :rtype:  str
    """
    if host:
        ipaddr = socket.gethostbyname(host)
    else:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.connect(('<broadcast>', 0))
                ipaddr = sock.getsockname()[0]
        except (Exception, BaseException):
            for url in ("https://ipecho.net/plain", "https://api.ipify.org"):
                try:
                    with urllib.request.urlopen(url=url).read().decode() as ipaddr:
                        if ipaddr:
                            break
                except (Exception, BaseException):
                    pass
            else:
                ipaddr = None
    return ipaddr


def is_localhost(host):
    """
    Determines if specified host/IP address canonically identifies localhost.

    :param host: Hostname/IPv4 (dotted-quad) address identifying host.
    :type  host: str

    :return: "Host is a canonical localhost address/name."
    :rtype:  bool
    """
    return host in ('localhost', '127.0.0.1', '0.0.0.0')


def quick_probe(host, port=None, timeout=2):
    """
    Performs a quick accessibility test for the specified host on a port.

    :param host:    Host name (FQDN) or IP address
    :type  host:    str
    :param port:    Port on which port accessibility test is performed
    :type  port:    int
    :param timeout: Timeout (sec) for accessibility test
    :type  timeout: float

    :return: "Host is accessible on port."
    :rtype:  bool
    """
    accessible = False
    try:
        parts = urlparse(host)
        host, *_port = (parts.netloc or parts.path).split(':')
        host = get_ipaddr(host)
        if port is None:
            port = int(_port[0]) if _port else parts.scheme or 'http'
        if isinstance(port, str):
            port = socket.getservbyname(port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            accessible = True
        except socket.timeout:
            pass
        finally:
            sock.close()
    except (BaseException, BaseException):
        pass
    return accessible
