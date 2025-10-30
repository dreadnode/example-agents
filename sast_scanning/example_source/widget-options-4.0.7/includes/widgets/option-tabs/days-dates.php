<?php

/**
 * Days and Dates Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Days & Dates Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_days($args)
{ ?>
    <li class="extended-widget-opts-tab-days">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-days" title="<?php _e('Days & Dates', 'widget-options'); ?>"><span class="dashicons dashicons-calendar-alt"></span> <span class="tabtitle"><?php _e('Days', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_days', 5);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
function widgetopts_tabcontent_days($args)
{
    global $widget_options;
    $days = array(
        'monday'    =>  __('Monday', 'widget-options'),
        'tuesday'   =>  __('Tuesday', 'widget-options'),
        'wednesday' =>  __('Wednesday', 'widget-options'),
        'thursday'  =>  __('Thursday', 'widget-options'),
        'friday'    =>  __('Friday', 'widget-options'),
        'saturday'  =>  __('Saturday', 'widget-options'),
        'sunday'    =>  __('Sunday', 'widget-options'),
    );
    $options_role = '';
    $options_dates = '';
    $from = '';
    $to = '';
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-days" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-days">
        <div class="extended-widget-opts-demo-feature">
            <div class="extended-widget-opts-demo-warning">
                <p class="widgetopts-unlock-features">
                    <span class="dashicons dashicons-lock"></span><br>
                    Unlock all Features<br>
                    <a href="https://widget-options.com/?utm_source=wordpressadmin&amp;utm_medium=widgettabs&amp;utm_campaign=widgetoptsprotab" class="button-primary" target="_blank">Learn More</a>
                </p>
            </div>
            <p>
                <strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
                <select class="widefat" readonly>
                    <option value="hide"><?php _e('Hide on checked days', 'widget-options'); ?></option>
                    <option value="show"><?php _e('Show on checked days', 'widget-options'); ?></option>
                </select>
            </p>
            <table class="form-table">
                <tbody>
                    <tr valign="top">
                        <td scope="row"><strong><?php _e('Days', 'widget-options'); ?></strong></td>
                        <td>&nbsp;</td>
                    </tr>
                    <?php foreach ($days as $key => $day) {
                        $checked = '';
                    ?>
                        <tr valign="top">
                            <td scope="row"><label><?php echo $day; ?></label></td>
                            <td>
                                <input type="checkbox" value="1" readonly />
                            </td>
                        </tr>
                    <?php } ?>
                </tbody>
            </table>
            <br />

            <p>
                <strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
                <select class="widefat" readonly>
                    <option value="hide"><?php _e('Hide on date range', 'widget-options'); ?></option>
                    <option value="show"><?php _e('Show on date range', 'widget-options'); ?></option>
                </select>
            </p>
            <table class="form-table">
                <tbody>
                    <tr valign="top">
                        <td scope="row"><strong><?php _e('From: ', 'widget-options'); ?></strong></td>
                        <td><input type="text" class="widefat extended-widget-opts-date" readonly /></td>
                    </tr>
                    <tr valign="top">
                        <td scope="row"><strong><?php _e('To: ', 'widget-options'); ?></strong></td>
                        <td><input type="text" class="widefat extended-widget-opts-date" readonly /></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_days'); ?>