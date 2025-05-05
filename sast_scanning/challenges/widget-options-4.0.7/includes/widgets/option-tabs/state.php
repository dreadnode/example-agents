<?php

/**
 * Roles Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Add Roles Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

/**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_state($args)
{ ?>
    <li class="extended-widget-opts-tab-roles">
        <a href="#extended-widget-opts-tab-<?php echo $args['id']; ?>-roles" title="<?php _e('Roles', 'widget-options'); ?>"><span class="dashicons dashicons-admin-users"></span> <span class="tabtitle"><?php _e('Roles', 'widget-options'); ?></span></a>
    </li>
<?php
}
add_action('extended_widget_opts_tabs', 'widgetopts_tab_state', 3);

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
function widgetopts_tabcontent_state($args)
{
    $roles          = get_editable_roles();
    $state   = '';
    if (isset($args['params']['roles']['state'])) {
        $state = $args['params']['roles']['state'];
    }
?>
    <div id="extended-widget-opts-tab-<?php echo $args['id']; ?>-roles" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-roles">
        <p class="widgetopts-subtitle"><?php _e('User Login State', 'widget-options'); ?></p>
        <p>
            <select class="widefat" name="<?php echo $args['namespace']; ?>[extended_widget_opts][roles][state]">
                <option value=""><?php _e('Select Visibility Option', 'widget-options'); ?></option>
                <option value="in" <?php if ($state == 'in') {
                                        echo 'selected="selected"';
                                    } ?>><?php _e('Show only for Logged-in Users', 'widget-options'); ?></option>
                <option value="out" <?php if ($state == 'out') {
                                        echo 'selected="selected"';
                                    } ?>><?php _e('Show only for Logged-out Users', 'widget-options'); ?></option>
            </select>
        </p>
        <p><small><?php _e('Restrict widget visibility for logged-in and logged-out users. ', 'widget-options'); ?></small></p>

        <div class="extended-widget-opts-demo-feature">
            <div class="extended-widget-opts-demo-warning">
                <p class="widgetopts-unlock-features">
                    <span class="dashicons dashicons-lock"></span><br>
                    Unlock all Features<br>
                    <a href="https://widget-options.com/?utm_source=wordpressadmin&amp;utm_medium=widgettabs&amp;utm_campaign=widgetoptsprotab" class="button-primary" target="_blank">Learn More</a>
                </p>
            </div>
            <p class="widgetopts-subtitle"><?php _e('User Roles', 'widget-options'); ?></p>
            <p>
                <strong><?php _e('Hide/Show', 'widget-options'); ?></strong>
                <select class="widefat" readonly>
                    <option value="hide"><?php _e('Hide on checked roles', 'widget-options'); ?></option>
                    <option value="show"><?php _e('Show on checked roles', 'widget-options'); ?></option>
                </select>
            </p>
            <div class="extended-widget-opts-inner-roles" style="max-height: 230px;padding: 5px;overflow:auto;">
                <table class="form-table">
                    <tbody>
                        <tr valign="top">
                            <td scope="row"><strong><?php _e('Roles', 'widget-options'); ?></strong></td>
                            <td>&nbsp;</td>
                        </tr>
                        <?php foreach ($roles as $role_name => $role_info) {
                            if (isset($args['params']) && isset($args['params']['roles'])) {
                                if (isset($args['params']['roles'][$role_name])) {
                                    $checked = 'checked="checked"';
                                } else {
                                    $checked = '';
                                }
                            } else {
                                $checked = '';
                            }
                        ?>
                            <tr valign="top">
                                <td scope="row"><label for="extended_widget_opts-<?php echo $args['id']; ?>-role-<?php echo $role_name; ?>"><?php echo $role_info['name']; ?></label></td>
                                <td>
                                    <input type="checkbox" value="1" readonly />
                                </td>
                            </tr>
                        <?php } ?>
                        <tr valign="top">
                            <td scope="row"><?php _e('Guests', 'widget-options'); ?></td>
                            <td>
                                <input type="checkbox" value="1" readonly />
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p><small><?php _e('Restrict widget visibility per user roles.', 'widget-options'); ?></small></p>
        </div>
    </div>
<?php
}
add_action('extended_widget_opts_tabcontent', 'widgetopts_tabcontent_state'); ?>