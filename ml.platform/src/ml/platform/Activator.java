package ml.platform;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;


public class Activator implements BundleActivator {
	Server server;

  public void start(BundleContext context)  {
	 System.out.println("PLUGIN STARTING.....");
	  
	  
	  AbstractHandler handler=new AbstractHandler()
		{
		   

			@Override
			public void handle(String arg0, Request arg1,
					HttpServletRequest request, HttpServletResponse response)
					throws IOException, ServletException {
				 response.setContentType("text/html");
			        response.setStatus(HttpServletResponse.SC_OK);
			        response.getWriter().println("<h1>Hello</h1>");
			        ((Request)request).setHandled(true);
				
			}
		};
		 
		 server = new Server(8080);
		server.setHandler(handler);
		try {
		server.start();
		}catch(Exception e)
		{	e.printStackTrace();}
		System.out.println("Server Started ");
		
  }

  
  public void stop(BundleContext context) throws Exception {
    server.stop();
  }

} 