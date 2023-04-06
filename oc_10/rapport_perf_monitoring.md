To control the performance of the model in production we use Application Insights. This will permit us to answer the question of "how many customers used the application. It will give us availability and performance monitoring. Go to the Azure portal and create the application Insights as in the figure below. 

<figure>
<a href="/img/AppIns.png"><img src="/img/AppIns.png"></a>
<figcaption>Figure 1: Application Insights.</figcaption>
</figure>

Check the resource mode ```Classic``` and click on ```Renew + create``` as in Figure 5. Then check the summary and click ```Create```. 

Once we created our Application Insights resources we can use the instrumentation key to configure our application insights sdk. We can copy it to the environment variables (.env file) to use it in our bot that will be created.


We also add application insights to monitor the bot's performance and monitor the problematic interactions (experiences) between the chatbot and the user. We use the final step to log in the application insights with AzureLogHandler. We track two details about a user's experience. The first thing to check is if the flight was satisfied with the bot's proposals and booked. And the second is if the customer was not satisfied with the bot's proposals. If the flight was booked we log it with the level INFO. A customer who is not satisfied is logged with the level ERROR.

```python
async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete the interaction and end the dialog."""
        booking_details = step_context.options
        
        if step_context.result:
            self.logger.setLevel(logging.INFO)
            self.logger.info('The flight is booked and the customer is satisfied.')

            return await step_context.end_dialog(booking_details)

        prop = {'custom_dimensions': booking_details.__dict__}
        
        self.logger.setLevel(logging.ERROR)
        self.logger.error('The customer was not satisfied about the bots proposals', extra=prop)
        
        
        return await step_context.end_dialog()
```

In addition, we trigger an alert if the user does not accept the bot's proposal three times within five minutes. The alert is created under Application Insights, Alerts. We click `Create` followed by `Create rule`, and set up our alert. Here is an example of an alert I made.

<figure>
<a href="/img/alert_ins.png"><img src="/img/alert_ins.png"></a>
<figcaption>Figure 2: Viewing the alert.</figcaption>
</figure>

As can be seen, the alert was triggered three times in five minutes. In the Error section, we have one alert. It is also possible to view the query that initiated the alert. We look for "not satisfied" in a trace message. The alert will be triggered if an item appears three times.

In future work, we can use application insights to:
 - monitor the number of requests that are made to our bot
 - monitor the number of errors that are made by our bot (ex: the user does not enter the correct date format)
 - monitor the number of users that are using our bot
 - monitor the number of times the user does not finalize the booking process
 - etc.